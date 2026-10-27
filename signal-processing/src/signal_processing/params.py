"""Frozen-parameter schema: written once by ``fit``, read by ``transform``."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from signal_processing.partition import PartitionRule

SCHEMA_VERSION = "1.0"
DEPTHS = ("5", "10")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Constants(_Frozen):
    """Design constants (chosen, not fitted). Stored with the params for reproducibility."""

    moisture_resolution: float = 0.001  # m3/m3, sensor reporting precision
    rain_resolution_mm: float = 0.1
    gap_reset_hours: int = 6  # state resets when > 6 h elapsed since the last valid value
    rain_window_hours: int = 3
    rain_threshold_mm: float = 0.5
    xcorr_window_hours: int = 72
    xcorr_max_lag_hours: int = 6
    xcorr_min_pairs: int = 48
    replay_window_hours: int = 12
    rough_window_hours: int = 24
    rough_min_periods: int = 18
    scale_floor: float = 0.0005  # half the moisture resolution; guards MAD = 0
    # CUSUM drift k_c = max(cusum_drift_min, quantile of |trailing mean of z|) on dry train
    # hours, so persistent model bias of a station does not accumulate without bound.
    cusum_drift_min: float = 0.5
    cusum_drift_quantile: float = 0.9
    cusum_drift_window_hours: int = 24
    cusum_drift_min_periods: int = 18
    cusum_threshold_quantile: float = 0.995
    # Max consecutive rain-hold hours = ceil(quantile of train hold-run lengths), clipped.
    hold_run_quantile: float = 0.99
    hold_min_hours: int = 6
    hold_max_hours: int = 48
    huber_c: float = 1.345
    tau_min_hours: float = 1.0
    tau_max_hours: float = 10_000.0
    min_fit_pairs: int = 500


class DrydownParams(_Frozen):
    """pred_t = theta_r + (theta_{t-1} - theta_r) * exp(-1/tau) + k * P_t."""

    theta_r: float
    tau_hours: float
    k_rain_per_mm: float
    n_fit: int


class DepthParams(_Frozen):
    drydown: DrydownParams
    innov_median: float
    innov_scale: float  # 1.4826 * MAD of the train innovation (floored)
    n_innov: int
    cusum_drift: float
    cusum_h: float  # suggested alarm threshold: train quantile of the CUSUM statistic


class StationParams(_Frozen):
    depths: dict[str, DepthParams]
    depth_gap_median: float
    depth_gap_scale: float
    cusum_max_hold_hours: int
    train_rows: int
    train_replay_windows: int
    train_replay_matches: int


class FrozenParams(_Frozen):
    schema_version: str = SCHEMA_VERSION
    input_sha256: str
    input_rows: int
    train_rows: int
    partition_source: str
    partition_rule: str
    partitions: tuple[PartitionRule, ...]
    constants: Constants
    stations: dict[str, StationParams]
