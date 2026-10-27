"""Causal signal features, computed per station on a regular hourly grid.

Each station's rows are placed on a complete hourly grid (hours with no row are
NaN; nothing is interpolated or filled). Every operation looks only backwards on
that grid: first differences, trailing windows ending at t, forward recursions.
A value at grid hour t therefore depends only on observations at hours <= t of
the same station. Results are read back at the original rows, in input order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from signal_processing.io import epoch_hours
from signal_processing.params import DEPTHS, Constants, DrydownParams, FrozenParams, StationParams

MAD_TO_SIGMA = 1.4826

FEATURE_COLUMNS = (
    *(f"sp_dtheta_dry_{d}" for d in DEPTHS),
    *(f"sp_innov_{d}" for d in DEPTHS),
    *(f"sp_innov_z_{d}" for d in DEPTHS),
    *(f"sp_cusum_{d}" for d in DEPTHS),
    "sp_depth_gap",
    "sp_depth_gap_z",
    "sp_depth_xcorr",
    "sp_replay_match",
    "sp_replay_run",
    *(f"sp_rough_{d}" for d in DEPTHS),
)
INTEGER_FEATURES = {"sp_replay_match": np.int8, "sp_replay_run": np.int32}


def moisture_column(depth: str) -> str:
    return f"SOIL_MOISTURE_{depth}"


# --------------------------------------------------------------------------- grid


@dataclass(frozen=True)
class Grid:
    """One station's signals on a complete hourly grid, plus each row's grid index."""

    positions: np.ndarray
    moisture: dict[str, np.ndarray]
    rain: np.ndarray


def build_grid(hours: np.ndarray, frame: pd.DataFrame) -> Grid:
    """Place rows (one station, unique hours, any order) on a gap-free hourly grid."""
    start = int(hours.min())
    positions = (hours - start).astype(np.int64)
    size = int(positions.max()) + 1

    def place(column: str) -> np.ndarray:
        out = np.full(size, np.nan)
        out[positions] = frame[column].to_numpy(dtype=float, na_value=np.nan)
        return out

    return Grid(
        positions=positions,
        moisture={d: place(moisture_column(d)) for d in DEPTHS},
        rain=place("P_CALC"),
    )


# ----------------------------------------------------------------- causal helpers


def lag(x: np.ndarray, hours: int) -> np.ndarray:
    """x[t - hours] on the grid (NaN before the start)."""
    out = np.full(x.shape, np.nan)
    if hours < len(x):
        out[hours:] = x[: len(x) - hours]
    return out


def trailing_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Sum over grid hours t-window+1..t; NaN if any value is NaN or history is short."""
    out = np.full(x.shape, np.nan)
    if len(x) >= window:
        out[window - 1 :] = sliding_window_view(x, window).sum(axis=1)
    return out


def trailing_mean(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Mean of the finite values over grid hours t-window+1..t (NaN below ``min_periods``)."""
    valid = np.isfinite(x)
    count = trailing_sum(valid.astype(float), window)
    total = trailing_sum(np.where(valid, x, 0.0), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[~(count >= min_periods)] = np.nan
    return mean


def robust_location_scale(values: np.ndarray, floor: float) -> tuple[float, float, int]:
    """Median and ``1.4826 * MAD`` (floored) of the finite values, plus their count."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite values to estimate a robust scale")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, max(MAD_TO_SIGMA * mad, floor), int(finite.size)


def rain_sum_tenths(rain: np.ndarray, c: Constants) -> np.ndarray:
    """Trailing rain total (incl. t) in integer tenths of a mm, so thresholds are exact."""
    return trailing_sum(np.rint(rain / c.rain_resolution_mm), c.rain_window_hours)


def rain_threshold_tenths(c: Constants) -> float:
    return float(round(c.rain_threshold_mm / c.rain_resolution_mm))


# ------------------------------------------------------------------- features 1-4


def dtheta_dry(theta: np.ndarray, rain_tenths: np.ndarray, c: Constants) -> np.ndarray:
    """Hourly difference where the trailing rain is below threshold, 0 where it rained."""
    diff = theta - lag(theta, 1)
    out = np.where(rain_tenths < rain_threshold_tenths(c), diff, 0.0)
    out[~np.isfinite(diff) | ~np.isfinite(rain_tenths)] = np.nan
    return out


def drydown_prediction(
    theta: np.ndarray, rain: np.ndarray, p: DrydownParams, max_elapsed: int
) -> np.ndarray:
    """Causal drydown prediction of theta_t from the last valid observation before t.

    With m = hours since that observation (1 <= m <= max_elapsed) the model is
    iterated m steps using the recorded rain of the intervening hours. Longer
    gaps reset the filter (NaN). Missing rain in the bridged hours gives NaN.
    """
    size = len(theta)
    a = float(np.exp(-1.0 / p.tau_hours))
    index = np.arange(size)
    last_valid = np.maximum.accumulate(np.where(np.isfinite(theta), index, -1))
    previous = np.concatenate(([-1], last_valid[:-1]))
    elapsed = index - previous
    pred = np.full(size, np.nan)
    rain_term = np.zeros(size)  # sum_{j<m} a^j * P[t-j]
    for m in range(1, max_elapsed + 1):
        rain_term = rain_term + a ** (m - 1) * lag(rain, m - 1)
        sel = (previous >= 0) & (elapsed == m)
        base = theta[previous[sel]]
        pred[sel] = p.theta_r + (base - p.theta_r) * a**m + p.k_rain_per_mm * rain_term[sel]
    return pred


def cusum(z: np.ndarray, hold: np.ndarray, drift: float, gap_reset: int) -> np.ndarray:
    """Two-sided CUSUM ``max(S+, S-)`` of z.

    NaN z gives NaN output and leaves the state untouched; more than ``gap_reset``
    hours since the last valid z resets both sums. Where ``hold`` is true (rain)
    the state is carried forward without an update. Sequential by nature.
    """
    out = np.full(len(z), np.nan)
    s_pos = s_neg = 0.0
    last = -(10**9)
    for i, (zi, held) in enumerate(zip(z.tolist(), hold.tolist(), strict=True)):
        if zi != zi:  # NaN
            continue
        if i - last > gap_reset:
            s_pos = s_neg = 0.0
        last = i
        if not held:
            s_pos = max(0.0, s_pos + zi - drift)
            s_neg = max(0.0, s_neg - zi - drift)
        out[i] = max(s_pos, s_neg)
    return out


def rain_hold(rain_tenths: np.ndarray, c: Constants) -> np.ndarray:
    """CUSUM hold mask: recent rain at/above threshold, or rain unknown."""
    return ~(rain_tenths < rain_threshold_tenths(c))


def hold_run_position(hold: np.ndarray) -> np.ndarray:
    """1-based position of each hour inside its consecutive hold run (0 outside runs)."""
    index = np.arange(1, len(hold) + 1)
    run_start = np.maximum.accumulate(np.where(~hold, index, 0))
    return np.where(hold, index - run_start, 0)


def capped_hold(hold: np.ndarray, max_hours: int) -> np.ndarray:
    """Keep only the first ``max_hours`` of each hold run.

    Injected rain (P_CALC is attackable) must not freeze the CUSUM indefinitely:
    beyond the longest plausible natural rain spell the statistic updates again.
    """
    return hold & (hold_run_position(hold) <= max_hours)


# ------------------------------------------------------------------ features 5-7


def depth_xcorr(sm5: np.ndarray, sm10: np.ndarray, c: Constants) -> np.ndarray:
    """Max trailing Pearson r between diff(SM5) at t-L and diff(SM10) at t, L = 0..max lag."""
    window = c.xcorr_window_hours
    d5 = sm5 - lag(sm5, 1)
    d10 = sm10 - lag(sm10, 1)
    best = np.full(len(sm5), np.nan)
    for shift in range(c.xcorr_max_lag_hours + 1):
        x = lag(d5, shift)
        valid = np.isfinite(x) & np.isfinite(d10)
        x0, y0 = np.where(valid, x, 0.0), np.where(valid, d10, 0.0)
        n = trailing_sum(valid.astype(float), window)
        sx, sy = trailing_sum(x0, window), trailing_sum(y0, window)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = trailing_sum(x0 * y0, window) - sx * sy / n
            var_x = trailing_sum(x0 * x0, window) - sx * sx / n
            var_y = trailing_sum(y0 * y0, window) - sy * sy / n
            r = cov / np.sqrt(var_x * var_y)
        ok = (n >= c.xcorr_min_pairs) & (var_x > 1e-12) & (var_y > 1e-12)
        best = np.fmax(best, np.where(ok, np.clip(r, -1.0, 1.0), np.nan))
    best[~np.isfinite(sm5) | ~np.isfinite(sm10)] = np.nan
    return best


def roughness(theta: np.ndarray, c: Constants) -> np.ndarray:
    """Trailing std (ddof=1) of the second difference; needs ``rough_min_periods`` values."""
    window = c.rough_window_hours
    second = theta - 2.0 * lag(theta, 1) + lag(theta, 2)
    out = np.full(len(theta), np.nan)
    if len(theta) < window:
        return out
    view = sliding_window_view(second, window)
    count = np.isfinite(view).sum(axis=1)
    total = np.where(np.isfinite(view), view, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        dev = np.where(np.isfinite(view), view - mean[:, None], 0.0)
        std = np.sqrt((dev * dev).sum(axis=1) / (count - 1))
    out[window - 1 :] = np.where(count >= c.rough_min_periods, std, np.nan)
    out[~np.isfinite(theta)] = np.nan
    return out


@dataclass(frozen=True)
class ReplayResult:
    match: np.ndarray
    run: np.ndarray
    windows: int
    matches: int


def replay_flags(sm5: np.ndarray, sm10: np.ndarray, rain: np.ndarray, c: Constants) -> ReplayResult:
    """Exact-repeat detector over trailing windows of (SM5, SM10, P) at sensor precision.

    A window is eligible when fully finite and neither moisture depth is flat in it
    (flat/frozen periods repeat naturally). Its integer key is looked up among all
    earlier eligible windows of the station; a hit sets ``match`` = 1. ``run`` counts
    consecutive matched hours.
    """
    size, window = len(sm5), c.replay_window_hours
    match = np.zeros(size, dtype=np.int8)
    run = np.zeros(size, dtype=np.int32)
    if size < window:
        return ReplayResult(match, run, 0, 0)
    stacked = np.column_stack(
        (
            np.rint(sm5 / c.moisture_resolution),
            np.rint(sm10 / c.moisture_resolution),
            np.rint(rain / c.rain_resolution_mm),
        )
    )
    views = sliding_window_view(stacked, (window, 3))[:, 0]
    finite = np.isfinite(views).all(axis=(1, 2))
    with np.errstate(invalid="ignore"):
        varied = (np.ptp(views[:, :, 0], axis=1) > 0) & (np.ptp(views[:, :, 1], axis=1) > 0)
    eligible = np.flatnonzero(finite & varied)
    keys = views[eligible].astype(np.int64)
    first_seen: dict[bytes, int] = {}
    for key, start in zip(keys, eligible.tolist(), strict=True):
        blob = key.tobytes()
        if blob in first_seen:
            match[start + window - 1] = 1
        else:
            first_seen[blob] = start
    streak = 0
    for i, hit in enumerate(match.tolist()):
        streak = streak + 1 if hit else 0
        run[i] = streak
    return ReplayResult(match, run, int(eligible.size), int(match.sum()))


# ---------------------------------------------------------------- orchestration


def station_features(grid: Grid, sp: StationParams, c: Constants) -> dict[str, np.ndarray]:
    """All features for one station, on its grid."""
    rain_tenths = rain_sum_tenths(grid.rain, c)
    hold = capped_hold(rain_hold(rain_tenths, c), sp.cusum_max_hold_hours)
    out: dict[str, np.ndarray] = {}
    for depth in DEPTHS:
        theta, dp = grid.moisture[depth], sp.depths[depth]
        innov = theta - drydown_prediction(theta, grid.rain, dp.drydown, c.gap_reset_hours)
        z = innov / dp.innov_scale
        out[f"sp_dtheta_dry_{depth}"] = dtheta_dry(theta, rain_tenths, c)
        out[f"sp_innov_{depth}"] = innov
        out[f"sp_innov_z_{depth}"] = z
        out[f"sp_cusum_{depth}"] = cusum(z, hold, dp.cusum_drift, c.gap_reset_hours)
        out[f"sp_rough_{depth}"] = roughness(theta, c)
    sm5, sm10 = grid.moisture["5"], grid.moisture["10"]
    gap = sm5 - sm10
    out["sp_depth_gap"] = gap
    out["sp_depth_gap_z"] = (gap - sp.depth_gap_median) / sp.depth_gap_scale
    out["sp_depth_xcorr"] = depth_xcorr(sm5, sm10, c)
    replay = replay_flags(sm5, sm10, grid.rain, c)
    out["sp_replay_match"] = replay.match
    out["sp_replay_run"] = replay.run
    return out


def transform(frame: pd.DataFrame, params: FrozenParams) -> pd.DataFrame:
    """Features for every input row: same rows, same keys, same order as ``frame``."""
    hours = epoch_hours(frame["time"])
    columns: dict[str, np.ndarray] = {
        name: np.zeros(len(frame), dtype=INTEGER_FEATURES[name])
        if name in INTEGER_FEATURES
        else np.full(len(frame), np.nan)
        for name in FEATURE_COLUMNS
    }
    groups = frame.groupby("station", sort=True).indices
    unknown = sorted(set(groups) - set(params.stations))
    if unknown:
        raise ValueError(f"stations without frozen params: {unknown}")
    for station, rows in groups.items():
        grid = build_grid(hours[rows], frame.iloc[rows])
        feats = station_features(grid, params.stations[station], params.constants)
        for name in FEATURE_COLUMNS:
            columns[name][rows] = feats[name][grid.positions]
    result = frame[["station", "time"]].reset_index(drop=True).copy()
    for name in FEATURE_COLUMNS:
        result[name] = columns[name]
    return result
