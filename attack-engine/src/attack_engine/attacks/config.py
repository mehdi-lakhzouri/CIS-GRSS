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


class AttackConfigError(Exception):
    """The attack configuration is missing, unreadable or invalid."""


class SpikeParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    magnitude: float = Field(gt=0.0)
    sign: SignPolicy = SignPolicy.RANDOM
    min_points: int = Field(default=1, ge=1)
    max_points: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _check_points(self) -> SpikeParameters:
        if self.min_points > self.max_points:
            raise ValueError("min_points must not exceed max_points")
        return self


class BiasParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    magnitude: float = Field(gt=0.0)
    sign: SignPolicy = SignPolicy.RANDOM


class DriftParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    final_magnitude: float = Field(gt=0.0)
    sign: SignPolicy = SignPolicy.RANDOM
    # Maximum absolute ramp slope in target units per hour; None disables the check.
    max_slope_per_hour: float | None = Field(default=None, gt=0.0)
    # The first observation of the ramp is left unchanged when True.
    hold_first: bool = True


class ReplayParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # Minimum age (hours) between the source's first observation and the window start.
    min_age_hours: int = Field(default=1, ge=1)


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
        return self


class AttackExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    global_seed: int = Field(ge=0)
    target_low: float = 0.0
    target_high: float = 0.70
    # Optional numerical quantum applied to candidate values only.
    quantum: float | None = Field(default=None, gt=0.0)
    spike: SpikeParameters | None = None
    bias: BiasParameters | None = None
    drift: DriftParameters | None = None
    replay: ReplayParameters | None = None
    stealth: StealthParameters | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> AttackExecutionConfig:
        if self.target_low > self.target_high:
            raise ValueError("target_low must not exceed target_high")
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
        config = config.model_copy(
            update={
                "stealth": config.stealth.model_copy(
                    update={
                        "calibration_path": (
                            path.parent / config.stealth.calibration_path
                        ).resolve()
                    }
                )
            }
        )
    return config
