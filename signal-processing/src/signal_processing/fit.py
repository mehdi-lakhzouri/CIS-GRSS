"""Train-only fitting of the frozen parameters.

Only rows of the train partition are used: the frame is cut to train rows
before any estimate, so validation/test values cannot influence a parameter.
Estimators are robust (Huber IRLS regression, median/MAD scales, quantiles).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_processing.features import (
    build_grid,
    capped_hold,
    cusum,
    drydown_prediction,
    hold_run_position,
    lag,
    rain_hold,
    rain_sum_tenths,
    replay_flags,
    robust_location_scale,
    trailing_mean,
)
from signal_processing.io import epoch_hours
from signal_processing.params import (
    DEPTHS,
    Constants,
    DepthParams,
    DrydownParams,
    FrozenParams,
    StationParams,
)
from signal_processing.partition import RULE_TEXT, PartitionRule, assign_partitions

_HUBER_ITERATIONS = 50


def huber_regression(design: np.ndarray, target: np.ndarray, c: float) -> np.ndarray:
    """Huber M-estimate by iteratively reweighted least squares (MAD residual scale)."""
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(_HUBER_ITERATIONS):
        resid = target - design @ beta
        scale = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))), 1e-9)
        u = np.abs(resid) / (c * scale)
        sw = np.sqrt(np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12)))
        updated = np.linalg.lstsq(design * sw[:, None], target * sw, rcond=None)[0]
        if np.allclose(updated, beta, rtol=1e-10, atol=1e-12):
            return updated
        beta = updated
    return beta


def fit_drydown(theta: np.ndarray, rain: np.ndarray, c: Constants) -> DrydownParams:
    """Fit theta_t = c0 + a * theta_{t-1} + k * P_t on consecutive train hours.

    The linear form is equivalent to the drydown model with a = exp(-1/tau) and
    theta_r = c0 / (1 - a). Constraints: k >= 0 and tau within the configured bounds.
    """
    prev, rain_t = lag(theta, 1), rain
    ok = np.isfinite(theta) & np.isfinite(prev) & np.isfinite(rain_t)
    n_fit = int(ok.sum())
    if n_fit < c.min_fit_pairs:
        raise ValueError(f"only {n_fit} train pairs to fit the drydown model")
    y, x, p = theta[ok], prev[ok], rain_t[ok]
    ones = np.ones_like(y)
    c0, a, k = huber_regression(np.column_stack((ones, x, p)), y, c.huber_c)
    if k < 0:
        (c0, a), k = huber_regression(np.column_stack((ones, x)), y, c.huber_c), 0.0
    a_lo, a_hi = np.exp(-1.0 / c.tau_min_hours), np.exp(-1.0 / c.tau_max_hours)
    if not a_lo <= a <= a_hi:
        a = float(np.clip(a, a_lo, a_hi))
        c0, k = huber_regression(np.column_stack((ones, p)), y - a * x, c.huber_c)
        if k < 0:
            (c0,), k = huber_regression(ones[:, None], y - a * x, c.huber_c), 0.0
    return DrydownParams(
        theta_r=float(c0 / (1.0 - a)),
        tau_hours=float(-1.0 / np.log(a)),
        k_rain_per_mm=float(k),
        n_fit=n_fit,
    )


def fit_cusum_drift(z: np.ndarray, hold: np.ndarray, c: Constants) -> float:
    """k_c = max(minimum, quantile of |trailing mean of z| over dry train hours)."""
    local = np.abs(trailing_mean(z, c.cusum_drift_window_hours, c.cusum_drift_min_periods))
    local = local[~hold & np.isfinite(local)]
    if local.size == 0:
        return c.cusum_drift_min
    return max(c.cusum_drift_min, float(np.quantile(local, c.cusum_drift_quantile)))


def fit_max_hold(hold: np.ndarray, c: Constants) -> int:
    """Longest plausible natural rain-hold run: ceil(train quantile), clipped to bounds."""
    position = hold_run_position(hold)
    run_ends = hold & ~np.concatenate((hold[1:], [False]))
    lengths = position[run_ends]
    if lengths.size == 0:
        return c.hold_min_hours
    estimate = int(np.ceil(np.quantile(lengths, c.hold_run_quantile)))
    return int(np.clip(estimate, c.hold_min_hours, c.hold_max_hours))


def fit_station(hours: np.ndarray, frame: pd.DataFrame, c: Constants) -> StationParams:
    """Fit one station from its train rows only."""
    grid = build_grid(hours, frame)
    hold = rain_hold(rain_sum_tenths(grid.rain, c), c)
    max_hold = fit_max_hold(hold, c)
    held = capped_hold(hold, max_hold)
    depths: dict[str, DepthParams] = {}
    for depth in DEPTHS:
        theta = grid.moisture[depth]
        drydown = fit_drydown(theta, grid.rain, c)
        innov = theta - drydown_prediction(theta, grid.rain, drydown, c.gap_reset_hours)
        median, scale, n_innov = robust_location_scale(innov, c.scale_floor)
        z = innov / scale
        drift = fit_cusum_drift(z, hold, c)
        stat = cusum(z, held, drift, c.gap_reset_hours)
        depths[depth] = DepthParams(
            drydown=drydown,
            innov_median=median,
            innov_scale=scale,
            n_innov=n_innov,
            cusum_drift=drift,
            cusum_h=float(np.quantile(stat[np.isfinite(stat)], c.cusum_threshold_quantile)),
        )
    gap_median, gap_scale, _ = robust_location_scale(
        grid.moisture["5"] - grid.moisture["10"], c.scale_floor
    )
    replay = replay_flags(grid.moisture["5"], grid.moisture["10"], grid.rain, c)
    return StationParams(
        depths=depths,
        depth_gap_median=gap_median,
        depth_gap_scale=gap_scale,
        cusum_max_hold_hours=max_hold,
        train_rows=len(frame),
        train_replay_windows=replay.windows,
        train_replay_matches=replay.matches,
    )


def fit_params(
    frame: pd.DataFrame,
    input_sha256: str,
    rules: tuple[PartitionRule, ...],
    partition_source: str,
    constants: Constants | None = None,
) -> FrozenParams:
    """Fit every station on the train partition and return frozen params.

    If the input carries a ``partition`` column it must agree with the rule.
    """
    c = constants or Constants()
    hours = epoch_hours(frame["time"])
    partitions = assign_partitions(hours, rules)
    if "partition" in frame.columns:
        declared = frame["partition"].astype(str).to_numpy()
        if np.any((partitions == "train") != (declared == "train")):
            raise ValueError("input 'partition' column disagrees with the train rule")
    is_train = partitions == "train"
    train = frame.loc[is_train].reset_index(drop=True)
    train_hours = hours[is_train]
    if train.empty:
        raise ValueError("no train rows in input")
    stations = {
        station: fit_station(train_hours[rows], train.iloc[rows], c)
        for station, rows in train.groupby("station", sort=True).indices.items()
    }
    return FrozenParams(
        input_sha256=input_sha256,
        input_rows=len(frame),
        train_rows=len(train),
        partition_source=partition_source,
        partition_rule=RULE_TEXT,
        partitions=rules,
        constants=c,
        stations=stations,
    )
