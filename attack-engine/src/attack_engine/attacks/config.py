"""Attack parameter configuration (Phases 3-4).

Magnitudes and policies live here, validated with Pydantic, not hardcoded inside
the transformations. All five families execute; missing Stealth parameters or a
calibration artifact are errors, never implicit skips.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from attack_engine.attacks.models import SignPolicy
from attack_engine.scheduling.models import AttackFamily

EXECUTABLE_FAMILIES = (
    AttackFamily.SPIKE,
    AttackFamily.BIAS,
    AttackFamily.DRIFT,
    AttackFamily.REPLAY,
    AttackFamily.STEALTH,
)

SecondaryColumn = Literal["SOIL_MOISTURE_10", "P_CALC"]


class AttackConfigError(Exception):
    """The attack configuration is missing, unreadable or invalid."""


PRIMARY_TARGET = "SOIL_MOISTURE_5"
# Variables an attack may target, with their physical units (never share a numeric offset
# across different units). Bounds are enforced per variable in attack_engine.attacks.multitarget.
MULTI_TARGET_COLUMNS = ("SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC")


def _validate_targets(
    target_columns: tuple[str, ...],
    secondary_magnitudes: dict[str, float],
    *,
    needs_magnitude: bool,
) -> None:
    if not target_columns:
        raise ValueError("target_columns must be non-empty")
    if len(set(target_columns)) != len(target_columns):
        raise ValueError("target_columns must be unique")
    unknown = [c for c in target_columns if c not in MULTI_TARGET_COLUMNS]
    if unknown:
        raise ValueError(f"unsupported target columns {unknown}; allowed {MULTI_TARGET_COLUMNS}")
    secondary = target_columns[1:]
    if needs_magnitude:
        missing = [c for c in secondary if c not in secondary_magnitudes]
        if missing:
            raise ValueError(f"secondary_magnitudes missing entries for {missing}")
    extra = [c for c in secondary_magnitudes if c not in secondary]
    if extra:
        raise ValueError(f"secondary_magnitudes has entries for non-secondary columns {extra}")
    if any(value <= 0.0 for value in secondary_magnitudes.values()):
        raise ValueError("secondary magnitudes must be positive")


class SpikeParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    magnitude: float = Field(gt=0.0)
    sign: SignPolicy = SignPolicy.RANDOM
    min_points: int = Field(default=1, ge=1)
    max_points: int = Field(default=3, ge=1)
    target_columns: tuple[str, ...] = (PRIMARY_TARGET,)
    # Per secondary column: its own, unit-aware spike magnitude (m3/m3 or mm).
    secondary_magnitudes: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_points(self) -> SpikeParameters:
        if self.min_points > self.max_points:
            raise ValueError("min_points must not exceed max_points")
        _validate_targets(self.target_columns, self.secondary_magnitudes, needs_magnitude=True)
        return self


class BiasParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    magnitude: float = Field(gt=0.0)
    sign: SignPolicy = SignPolicy.RANDOM
    target_columns: tuple[str, ...] = (PRIMARY_TARGET,)
    secondary_magnitudes: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_targets(self) -> BiasParameters:
        _validate_targets(self.target_columns, self.secondary_magnitudes, needs_magnitude=True)
        return self


class DriftParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    final_magnitude: float = Field(gt=0.0)
    sign: SignPolicy = SignPolicy.RANDOM
    # Maximum absolute ramp slope in target units per hour; None disables the check.
    max_slope_per_hour: float | None = Field(default=None, gt=0.0)
    # The first observation of the ramp is left unchanged when True.
    hold_first: bool = True
    target_columns: tuple[str, ...] = (PRIMARY_TARGET,)
    # Per secondary column: its own final deviation magnitude for the ramp.
    secondary_magnitudes: dict[str, float] = Field(default_factory=dict)
    secondary_max_slopes_per_hour: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_targets(self) -> DriftParameters:
        _validate_targets(self.target_columns, self.secondary_magnitudes, needs_magnitude=True)
        for column, slope in self.secondary_max_slopes_per_hour.items():
            if column not in self.target_columns[1:] or slope <= 0:
                raise ValueError("secondary slope limits must be positive and name a secondary")
        return self


class ReplayParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # Minimum age (hours) between the source's first observation and the window start.
    min_age_hours: int = Field(default=1, ge=1)
    # Replay copies the source values for every selected variable; no magnitude is needed.
    target_columns: tuple[str, ...] = (PRIMARY_TARGET,)

    @model_validator(mode="after")
    def _check_targets(self) -> ReplayParameters:
        _validate_targets(self.target_columns, {}, needs_magnitude=False)
        return self


class MissingContextPolicy(StrEnum):
    REJECT_WINDOW = "reject_window"
    SKIP_OBSERVATION = "skip_observation"


ContextFeature = Literal["SOIL_MOISTURE_10", "P_CALC", "SOIL_TEMP_5", "T_HR_AVG"]


class StealthParameters(BaseModel):
    """Experimental constraints, not detector thresholds or guarantees of evasion."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    magnitude_min: float = Field(default=0.005, gt=0.0)
    magnitude_max: float = Field(default=0.015, gt=0.0)
    sign: SignPolicy = SignPolicy.RANDOM
    residual_tolerance_multiplier: float = Field(default=1.0, gt=0.0)
    # Rate of change of the injected offset, in m3/m3 per elapsed hour.
    max_change_rate: float = Field(default=0.003, gt=0.0)
    # Rate bound on the emitted measurement itself (natural changes plus the offset).
    max_step_change: float = Field(default=0.05, gt=0.0)
    # Total variation of the injected offset, starting from zero, across an event.
    max_cumulative_drift: float = Field(default=0.08, gt=0.0)
    # Minimum mean absolute error over ALL reserved positions, after quantization.
    min_attack_impact: float = Field(default=0.002, gt=0.0)
    required_context_features: tuple[ContextFeature, ...] = (
        "SOIL_MOISTURE_10",
        "P_CALC",
        "SOIL_TEMP_5",
        "T_HR_AVG",
    )
    min_history_hours: int = Field(default=24, ge=1)
    calibration_path: Path
    missing_context_policy: MissingContextPolicy = MissingContextPolicy.REJECT_WINDOW
    max_candidate_attempts: int = Field(default=8, ge=1, le=1000)
    # The MVP never attacks its calibration partition, and has no train-prefix mode.
    attack_partitions: tuple[Literal["validation", "test"], ...] = ("validation", "test")
    ridge_alpha: float = Field(default=1.0, gt=0.0)
    residual_quantile: float = Field(default=0.99, gt=0.0, lt=1.0)
    residual_floor: float = Field(default=0.001, gt=0.0)
    min_training_rows: int = Field(default=240, ge=2)

    # Coordinated multi-variable strategy. Empty (the default) is single-target stealth,
    # identical to the baseline. When set, the soil-moisture channel is unchanged and these
    # context variables are coordinated with it; their couplings are fitted on train.
    secondary_columns: tuple[SecondaryColumn, ...] = ()
    target_columns: tuple[str, ...] = (PRIMARY_TARGET,)

    @model_validator(mode="before")
    @classmethod
    def _resolve_strategy_targets(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            secondary = tuple(data.get("secondary_columns", ()))
            targets = tuple(data.get("target_columns", (PRIMARY_TARGET, *secondary)))
            expected = tuple(c for c in targets if c != PRIMARY_TARGET)
            if secondary and secondary != expected:
                raise ValueError("secondary_columns conflicts with target_columns")
            data["target_columns"] = targets
            data["secondary_columns"] = expected
        return data

    # SOIL_MOISTURE_10 depth coupling (slope calibrated on train, clamped and lagged).
    depth_coupling_max_ratio: float = Field(default=1.0, gt=0.0)
    depth_coupling_lag_hours: int = Field(default=1, ge=0)
    depth_low: float = Field(default=0.0, ge=0.0, le=0.7)
    depth_high: float = Field(default=0.70, gt=0.0, le=0.7)
    depth_max_step_change: float = Field(default=0.05, gt=0.0)

    # P_CALC rainfall injection: a bounded pulse placed before a moisture rise, never fabricated
    # into a missing (NaN) rainfall slot.
    rainfall_lag_hours: int = Field(default=2, ge=0)
    rainfall_pulse_mm: float = Field(default=1.0, gt=0.0)
    rainfall_trigger: float = Field(default=0.003, gt=0.0)
    rainfall_max_mm: float = Field(default=200.0, gt=0.0, le=200.0)

    @model_validator(mode="after")
    def _check_stealth(self) -> StealthParameters:
        if self.magnitude_min > self.magnitude_max:
            raise ValueError("magnitude_min must not exceed magnitude_max")
        if self.min_attack_impact > min(self.magnitude_max, self.max_cumulative_drift):
            raise ValueError("min_attack_impact exceeds the possible requested impact")
        for name in ("required_context_features", "attack_partitions"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be non-empty and unique")
        if len(set(self.secondary_columns)) != len(self.secondary_columns):
            raise ValueError("secondary_columns must be unique")
        if self.depth_low >= self.depth_high:
            raise ValueError("depth_low must be below depth_high")
        _validate_targets(self.target_columns, {}, needs_magnitude=False)
        return self


class VariableConstraints(BaseModel):
    """Limits in this variable's units; omitted bounds use the physical defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    low: float | None = None
    high: float | None = None
    quantum: float | None = Field(default=None, gt=0.0)
    max_step_per_hour: float | None = Field(default=None, gt=0.0)
    max_offset_rate_per_hour: float | None = Field(default=None, gt=0.0)


class CrossVariableConstraint(BaseModel):
    """Require lower <= left - scale * right <= upper, in left's units."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    left: Literal["SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC"]
    right: Literal["SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC"]
    scale: float = 1.0
    lower: float
    upper: float
    quantity: Literal["values", "offsets"] = "values"

    @model_validator(mode="after")
    def _check_range(self) -> CrossVariableConstraint:
        if self.left == self.right or self.lower > self.upper:
            raise ValueError("relationship needs distinct variables and ordered limits")
        return self


class AttackExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    global_seed: int = Field(ge=0)
    target_low: float = Field(default=0.0, ge=0.0, le=0.7)
    target_high: float = Field(default=0.70, ge=0.0, le=0.7)
    # Optional numerical quantum applied to candidate values only.
    quantum: float | None = Field(default=None, gt=0.0)
    variable_constraints: dict[str, VariableConstraints] = Field(default_factory=dict)
    relationships: tuple[CrossVariableConstraint, ...] = ()
    spike: SpikeParameters | None = None
    bias: BiasParameters | None = None
    drift: DriftParameters | None = None
    replay: ReplayParameters | None = None
    stealth: StealthParameters | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> AttackExecutionConfig:
        if self.target_low > self.target_high:
            raise ValueError("target_low must not exceed target_high")
        for column, limits in self.variable_constraints.items():
            if column not in MULTI_TARGET_COLUMNS:
                raise ValueError(f"unsupported constrained variable {column}")
            low = limits.low if limits.low is not None else 0.0
            high = (
                limits.high if limits.high is not None else (200.0 if column == "P_CALC" else 0.7)
            )
            if low < 0 or low > high or high > (200.0 if column == "P_CALC" else 0.7):
                raise ValueError(f"invalid physical bounds for {column}")
        return self

    def parameters_for(self, family: AttackFamily) -> BaseModel:
        """Return the validated parameters for ``family`` or raise a helpful error."""
        if family not in EXECUTABLE_FAMILIES:
            supported = ", ".join(sorted(f.value for f in EXECUTABLE_FAMILIES))
            raise AttackConfigError(
                f"attack family {family.value!r} is not executable in this phase "
                f"(supported: {supported})"
            )
        params = getattr(self, family.value)
        if params is None:
            raise AttackConfigError(f"no parameters configured for attack family {family.value!r}")
        return params

    def has_parameters(self, family: AttackFamily) -> bool:
        return family in EXECUTABLE_FAMILIES and getattr(self, family.value) is not None

    def target_columns_for(self, family: AttackFamily) -> tuple[str, ...]:
        """Resolve the variables a family attacks (primary first), for any family."""
        columns = getattr(self.parameters_for(family), "target_columns", None)
        return tuple(columns) if columns else (PRIMARY_TARGET,)


def load_attack_config(path: Path) -> AttackExecutionConfig:
    path = Path(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise AttackConfigError(f"attack configuration not found: {path}") from None
    except OSError as exc:
        raise AttackConfigError(f"cannot read attack configuration {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise AttackConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AttackConfigError(f"attack configuration {path} must contain a mapping")
    try:
        config = AttackExecutionConfig.model_validate(data)
    except ValidationError as exc:
        raise AttackConfigError(f"invalid attack configuration {path}:\n{exc}") from exc
    if config.stealth is not None and not config.stealth.calibration_path.is_absolute():
        resolved = (path.parent / config.stealth.calibration_path).resolve()
        config = config.model_copy(
            update={"stealth": config.stealth.model_copy(update={"calibration_path": resolved})}
        )
    return config
