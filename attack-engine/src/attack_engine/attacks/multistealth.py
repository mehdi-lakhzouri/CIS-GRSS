"""Coordinated multi-variable stealth: one event modifies the soil-moisture target and,
as an explicit attacker capability, one or more coupled context variables.

The soil-moisture (``SOIL_MOISTURE_5``) channel reuses the existing single-variable stealth
transformation and its frozen train calibration unchanged. The secondary variables are driven
by lightweight couplings fitted on the clean train partition only:

- ``SOIL_MOISTURE_10`` follows the 5 cm injected offset through a per-station depth slope,
  lagged and continuity-limited (deeper moisture responds, damped and delayed);
- ``P_CALC`` receives a bounded rain pulse placed *before* a moisture rise, never exceeding a
  per-station train rainfall ceiling and never fabricated into a missing rainfall slot.

Missing values are preserved, physical bounds enforced, and no constraint is relaxed to
consume a quota. This module is a strategy of Stealth, with every changed secondary cell
charged by the shared executor under the configured row/cell budget policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from attack_engine.attacks import base, multitarget
from attack_engine.attacks.config import StealthParameters
from attack_engine.attacks.models import (
    AttackRequest,
    AttackSettings,
    ColumnModification,
    TargetBounds,
)
from attack_engine.dataset import temporal
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan

MOISTURE_10 = "SOIL_MOISTURE_10"
RAINFALL = "P_CALC"


class StationCoupling(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # Slope of SOIL_MOISTURE_10 on SOIL_MOISTURE_5, clamped to [0, max_ratio]; 0 when
    # too few complete train rows (the 10 cm channel then stays genuine for that station).
    depth_slope: float = Field(ge=0.0)
    depth_rows: int = Field(ge=0)
    # Per-station rainfall ceiling (mm): a high train quantile of non-zero P_CALC, capped at
    # the configured maximum. 0 when the station has no rain in train (no pulse is possible).
    rainfall_ceiling_mm: float = Field(ge=0.0)
    rainfall_rows: int = Field(ge=0)


class SecondaryCalibration(BaseModel):
    """Per-station secondary couplings, fitted on the clean train partition only."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    secondary_columns: tuple[str, ...]
    partition_start: str
    partition_end: str
    stations: dict[str, StationCoupling]

    def coupling(self, station: str) -> StationCoupling:
        return self.stations.get(
            station,
            StationCoupling(
                depth_slope=0.0, depth_rows=0, rainfall_ceiling_mm=0.0, rainfall_rows=0
            ),
        )


def fit_secondary_calibration(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    partitions: PartitionPlan,
    params: StealthParameters,
) -> SecondaryCalibration:
    """Fit per-station depth slope and rainfall ceiling on the train partition only."""
    train = next((p for p in partitions.partitions if p.name == "train"), None)
    if train is None:
        raise ValueError("multi-stealth calibration requires an explicit train partition")
    times = temporal.to_utc(frame[schema.timestamps.column])
    train_mask = temporal.assign_partitions(times, partitions, schema.timestamps.interval).eq(
        "train"
    )
    train_frame = frame.loc[train_mask]
    target = schema.target_column
    quantile = params.residual_quantile
    stations: dict[str, StationCoupling] = {}
    for station, group in train_frame.groupby(schema.station_column, sort=True):
        depth_slope, depth_rows = _fit_depth_slope(group, target, params)
        ceiling, rain_rows = _fit_rainfall_ceiling(group, params, quantile)
        stations[str(station)] = StationCoupling(
            depth_slope=depth_slope,
            depth_rows=depth_rows,
            rainfall_ceiling_mm=ceiling,
            rainfall_rows=rain_rows,
        )
    return SecondaryCalibration(
        secondary_columns=params.secondary_columns,
        partition_start=train.start.isoformat(),
        partition_end=train.end.isoformat(),
        stations=stations,
    )


def _fit_depth_slope(
    group: pd.DataFrame, target: str, params: StealthParameters
) -> tuple[float, int]:
    if MOISTURE_10 not in group:
        return 0.0, 0
    x = group[target].to_numpy(dtype="float64")
    y = group[MOISTURE_10].to_numpy(dtype="float64")
    mask = np.isfinite(x) & np.isfinite(y)
    rows = int(mask.sum())
    if rows < params.min_training_rows:
        return 0.0, rows
    x, y = x[mask], y[mask]
    variance = float(np.var(x))
    if variance <= np.finfo(float).eps:
        return 0.0, rows
    slope = float(np.cov(x, y, bias=True)[0, 1] / variance)
    return float(np.clip(slope, 0.0, params.depth_coupling_max_ratio)), rows


def _fit_rainfall_ceiling(
    group: pd.DataFrame, params: StealthParameters, quantile: float
) -> tuple[float, int]:
    if RAINFALL not in group:
        return 0.0, 0
    rain = group[RAINFALL].to_numpy(dtype="float64")
    positive = rain[np.isfinite(rain) & (rain > 0.0)]
    if positive.size == 0:
        return 0.0, 0
    ceiling = float(np.quantile(positive, quantile))
    return min(ceiling, params.rainfall_max_mm), int(positive.size)


def coordinate_secondary(
    request: AttackRequest,
    params: StealthParameters,
    offset5: np.ndarray,
    settings: AttackSettings,
) -> tuple[ColumnModification, ...]:
    """Secondary-variable modifications coordinated with the soil-moisture offset.

    Returns an empty tuple when no multi-stealth context is supplied or no secondary variable
    changes. The soil-moisture channel is produced by the unchanged single-variable stealth.
    """
    context = request.multistealth_context
    if context is None:
        return ()
    coupling = context.calibration.coupling(request.window.station)
    extra: list[ColumnModification] = []
    for column in params.secondary_columns:
        modification = _secondary_column(column, request, offset5, coupling, params, settings)
        if modification is not None:
            extra.append(modification)
    return tuple(extra)


def _secondary_column(
    column: str,
    request: AttackRequest,
    offset5: np.ndarray,
    coupling: StationCoupling,
    params: StealthParameters,
    settings: AttackSettings,
) -> ColumnModification | None:
    clean = np.asarray(request.multistealth_context.secondary_clean[column], dtype="float64")
    elapsed = np.asarray(request.elapsed_hours, dtype=float)
    if column == MOISTURE_10:
        candidate = _depth_candidate(clean, offset5, coupling, params, elapsed)
        bounds = TargetBounds(params.depth_low, params.depth_high)
    elif column == RAINFALL:
        candidate = _rainfall_candidate(clean, offset5, coupling, params, elapsed)
        bounds = TargetBounds(0.0, params.rainfall_max_mm)
    else:  # pragma: no cover - guarded by the config's Literal type
        return None

    resolved = multitarget.settings_for(column, settings)
    low, high = max(bounds.low, resolved.bounds.low), min(bounds.high, resolved.bounds.high)
    if low > high:
        return None
    bounds = TargetBounds(low, high)
    column_settings = AttackSettings(bounds=bounds, quantum=resolved.quantum)
    final, changed, _ = base.finalize_candidate(clean, candidate, column_settings)
    if not changed.any():
        return None
    return ColumnModification(
        column=column,
        row_labels=request.window.row_labels,
        clean_values=tuple(float(value) for value in clean),
        candidate_values=tuple(float(value) for value in final),
        changed_mask=tuple(bool(flag) for flag in changed),
    )


def _depth_candidate(
    clean: np.ndarray,
    offset5: np.ndarray,
    coupling: StationCoupling,
    params: StealthParameters,
    elapsed: np.ndarray | None = None,
) -> np.ndarray:
    """Lagged, continuity-limited depth response; missing 10 cm values stay genuine."""
    candidate = clean.copy()
    if coupling.depth_slope <= 0.0:
        return candidate
    lag = params.depth_coupling_lag_hours
    elapsed = np.arange(clean.size, dtype=float) if elapsed is None else elapsed
    previous = None
    for index in range(clean.size):
        if not np.isfinite(clean[index]):
            previous = None  # a gap resets continuity; never fabricate a missing value
            continue
        source_index = int(np.searchsorted(elapsed, elapsed[index] - lag, side="right")) - 1
        source = offset5[source_index] if source_index >= 0 else 0.0
        target_offset = coupling.depth_slope * source
        value = float(np.clip(clean[index] + target_offset, params.depth_low, params.depth_high))
        if previous is not None:
            step = params.depth_max_step_change * (elapsed[index] - elapsed[index - 1])
            value = float(np.clip(value, previous - step, previous + step))
        candidate[index] = value
        previous = value
    return candidate


def _rainfall_candidate(
    clean: np.ndarray,
    offset5: np.ndarray,
    coupling: StationCoupling,
    params: StealthParameters,
    elapsed: np.ndarray | None = None,
) -> np.ndarray:
    """A bounded rain pulse placed before each moisture rise; never fills a missing slot."""
    candidate = clean.copy()
    ceiling = min(params.rainfall_max_mm, coupling.rainfall_ceiling_mm)
    if ceiling <= 0.0:
        return candidate
    lag = params.rainfall_lag_hours
    elapsed = np.arange(clean.size, dtype=float) if elapsed is None else elapsed
    rises = np.diff(offset5, prepend=offset5[:1])
    for index in range(clean.size):
        if rises[index] <= params.rainfall_trigger:
            continue
        pulse_time = elapsed[index] - lag
        position = int(np.searchsorted(elapsed, pulse_time))
        if (
            position >= clean.size
            or elapsed[position] != pulse_time
            or not np.isfinite(clean[position])
        ):
            continue  # no genuine rainfall slot to carry the injected pulse
        headroom = ceiling - candidate[position]
        if headroom <= 0.0:
            continue
        candidate[position] = candidate[position] + min(params.rainfall_pulse_mm, headroom)
    return candidate
