"""Station ridge references fitted exclusively on the authorized clean train partition.

No detector is trained here. Missing measurements are excluded, never replaced by zero.
The artifact is deterministic JSON; its SHA-256 excludes its own digest field.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from attack_engine.attacks.config import StealthParameters
from attack_engine.dataset import temporal
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan

DERIVED_FEATURES = (
    "target_lag_1",
    "target_history_mean",
    "season_sin",
    "season_cos",
    "hour_sin",
    "hour_cos",
)


class CalibrationError(ValueError):
    """Invalid, incompatible, missing or unauthorized contextual calibration."""


def time_features(time: datetime) -> tuple[float, ...]:
    """Seasonal phase and UTC end-of-hour phase; no future measurements."""
    stamp = pd.Timestamp(time)
    day = stamp.day_of_year - 1 + (stamp.hour + stamp.minute / 60) / 24
    season = 2 * np.pi * day / (366 if stamp.is_leap_year else 365)
    hour = 2 * np.pi * (stamp.hour + stamp.minute / 60) / 24
    return float(np.sin(season)), float(np.cos(season)), float(np.sin(hour)), float(np.cos(hour))


def feature_vector(
    measurements: tuple[float, ...], history: tuple[float, ...], time: datetime
) -> np.ndarray:
    return np.asarray((*measurements, history[-1], float(np.mean(history)), *time_features(time)))


class StationReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    intercept: float
    coefficients: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    residual_tolerance: float = Field(gt=0.0)
    training_rows: int = Field(ge=1)
    eligible_rows: int = Field(ge=2)
    first_training_time: AwareDatetime
    last_training_time: AwareDatetime

    @model_validator(mode="after")
    def _dimensions(self) -> StationReference:
        if not len(self.coefficients) == len(self.feature_means) == len(self.feature_scales):
            raise ValueError("coefficient and normalization dimensions differ")
        if not self.coefficients or min(self.feature_scales) <= 0:
            raise ValueError("normalization scales must be positive")
        return self

    def predict(self, features: np.ndarray) -> float:
        normalized = (features - np.asarray(self.feature_means)) / np.asarray(self.feature_scales)
        return self.intercept + float(normalized @ np.asarray(self.coefficients))


class ContextCalibration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    method: Literal["station_standardized_ridge"] = "station_standardized_ridge"
    training_partition: Literal["train"] = "train"
    partition_start: AwareDatetime
    partition_end: AwareDatetime
    target_column: str
    context_features: tuple[str, ...]
    feature_names: tuple[str, ...]
    history_hours: int = Field(ge=1)
    interval_hours: float = Field(gt=0.0)
    ridge_alpha: float = Field(gt=0.0)
    residual_quantile: float = Field(gt=0.0, lt=1.0)
    residual_floor: float = Field(gt=0.0)
    training_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    station_models: dict[str, StationReference]
    unavailable_stations: dict[str, str]
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_models(self) -> ContextCalibration:
        if self.partition_start >= self.partition_end or not self.station_models:
            raise ValueError("calibration needs a valid train interval and station models")
        if self.feature_names != (*self.context_features, *DERIVED_FEATURES):
            raise ValueError("unexpected feature order")
        samples = self.history_hours / self.interval_hours
        if not samples.is_integer() or samples < 1:
            raise ValueError("calibration history is not a whole number of intervals")
        for model in self.station_models.values():
            if len(model.coefficients) != len(self.feature_names):
                raise ValueError("model dimensions do not match features")
            if not (
                self.partition_start
                < model.first_training_time
                <= model.last_training_time
                <= self.partition_end
            ):
                raise ValueError("station calibration observations are outside train")
        return self

    def computed_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"calibration_sha256"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def history_samples(self) -> int:
        return round(self.history_hours / self.interval_hours)

    def require_compatible(
        self, params: StealthParameters, schema: DatasetSchema, partitions: PartitionPlan
    ) -> None:
        train = next((p for p in partitions.partitions if p.name == "train"), None)
        if (
            train is None
            or (train.start, train.end) != (self.partition_start, self.partition_end)
            or self.context_features != params.required_context_features
            or self.history_hours != params.min_history_hours
            or self.target_column != schema.target_column
            or self.interval_hours != schema.timestamps.interval.total_seconds() / 3600
            or (self.ridge_alpha, self.residual_quantile, self.residual_floor)
            != (params.ridge_alpha, params.residual_quantile, params.residual_floor)
        ):
            raise CalibrationError("calibration does not match train, schema or feature settings")


def training_features(
    station_frame: pd.DataFrame, schema: DatasetSchema, params: StealthParameters
) -> pd.DataFrame:
    """Causal lag features; reset history at missing targets and temporal gaps."""
    times = temporal.to_utc(station_frame[schema.timestamps.column])
    target = station_frame[schema.target_column].astype("float64")
    step = schema.timestamps.interval
    samples = round(params.min_history_hours / (step.total_seconds() / 3600))
    previous = target.shift(1)
    breaks = times.diff().ne(step) | ~np.isfinite(target) | ~np.isfinite(previous)
    groups = breaks.cumsum()
    features = station_frame.reindex(columns=list(params.required_context_features)).astype(float)
    features = features.copy()
    features["target_lag_1"] = target.groupby(groups, sort=False).shift(1)
    features["target_history_mean"] = target.groupby(groups, sort=False).transform(
        lambda values: values.shift(1).rolling(samples, min_periods=samples).mean()
    )
    features[list(DERIVED_FEATURES[2:])] = np.asarray(
        [time_features(time.to_pydatetime()) for time in times]
    )
    return features


def fit_calibration(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    partitions: PartitionPlan,
    params: StealthParameters,
) -> ContextCalibration:
    train = next((p for p in partitions.partitions if p.name == "train"), None)
    if train is None:
        raise CalibrationError("an explicitly named train partition is required")
    interval_hours = schema.timestamps.interval.total_seconds() / 3600
    samples = params.min_history_hours / interval_hours
    if not samples.is_integer() or samples < 1:
        raise CalibrationError("history hours must be a positive whole number of intervals")
    times = temporal.to_utc(frame[schema.timestamps.column])
    mask = temporal.assign_partitions(times, partitions, schema.timestamps.interval).eq("train")
    # Slice BEFORE feature generation: validation/test observations are never model inputs.
    columns = [
        schema.station_column,
        schema.timestamps.column,
        schema.target_column,
        *params.required_context_features,
    ]
    training = frame.loc[mask].reindex(columns=columns).copy()
    training[schema.timestamps.column] = times.loc[mask]
    training = training.sort_values(
        [schema.station_column, schema.timestamps.column], kind="stable"
    )
    if training.empty:
        raise CalibrationError("no observations in the authorized train partition")
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(training, index=False).to_numpy().tobytes()
    ).hexdigest()
    models: dict[str, StationReference] = {}
    unavailable: dict[str, str] = {}
    for station, group in training.groupby(schema.station_column, sort=True):
        x = training_features(group, schema, params).to_numpy(dtype=float)
        y = group[schema.target_column].to_numpy(dtype=float)
        eligible = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if int(eligible.sum()) < params.min_training_rows:
            unavailable[str(station)] = f"{int(eligible.sum())} complete causal training rows"
            continue
        x, y = x[eligible], y[eligible]
        means, scales = x.mean(axis=0), x.std(axis=0)
        scales = np.where(scales > np.finfo(float).eps, scales, 1.0)
        z = (x - means) / scales
        intercept = float(y.mean())
        coefficients = np.linalg.solve(
            z.T @ z + params.ridge_alpha * np.eye(z.shape[1]), z.T @ (y - intercept)
        )
        residuals = y - (intercept + z @ coefficients)
        tolerance = max(
            params.residual_floor, float(np.quantile(np.abs(residuals), params.residual_quantile))
        )
        eligible_times = group.loc[eligible, schema.timestamps.column]
        models[str(station)] = StationReference(
            intercept=intercept,
            coefficients=tuple(coefficients),
            feature_means=tuple(means),
            feature_scales=tuple(scales),
            residual_tolerance=tolerance,
            training_rows=len(group),
            eligible_rows=len(y),
            first_training_time=eligible_times.iloc[0].to_pydatetime(),
            last_training_time=eligible_times.iloc[-1].to_pydatetime(),
        )
    if not models:
        raise CalibrationError(f"no station has enough complete training context: {unavailable}")
    artifact = ContextCalibration(
        partition_start=train.start,
        partition_end=train.end,
        target_column=schema.target_column,
        context_features=params.required_context_features,
        feature_names=(*params.required_context_features, *DERIVED_FEATURES),
        history_hours=params.min_history_hours,
        interval_hours=interval_hours,
        ridge_alpha=params.ridge_alpha,
        residual_quantile=params.residual_quantile,
        residual_floor=params.residual_floor,
        training_sha256=fingerprint,
        station_models=models,
        unavailable_stations=unavailable,
        calibration_sha256="0" * 64,
    )
    return artifact.model_copy(update={"calibration_sha256": artifact.computed_sha256()})


def load_calibration(path: Path) -> ContextCalibration:
    try:
        artifact = ContextCalibration.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise CalibrationError(f"cannot load calibration {path}: {exc}") from exc
    if artifact.computed_sha256() != artifact.calibration_sha256:
        raise CalibrationError("calibration SHA-256 mismatch")
    return artifact


def save_calibration(artifact: ContextCalibration, path: Path) -> None:
    """Explicit artifact write, never a dataset export; no overwrite of different content."""
    if path.suffix.lower() != ".json":
        raise CalibrationError("calibration output must be a separate .json file")
    if path.exists():
        if load_calibration(path) == artifact:
            return
        raise CalibrationError(f"calibration already exists with different content: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    artifact.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False
                )
                + "\n"
            )
    except OSError as exc:
        raise CalibrationError(f"cannot save calibration {path}: {exc}") from exc
