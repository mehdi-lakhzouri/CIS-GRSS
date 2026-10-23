"""Scheduling configuration: YAML parsed into a validated, immutable model.

Invalid durations, negative or out-of-range budgets, empty or duplicated family
lists, and weights for disabled families are all rejected rather than silently
accepted.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from attack_engine.scheduling.models import AttackFamily


class ConfigError(Exception):
    """The scheduling configuration is missing, unreadable or invalid."""


class InsufficientPolicy(StrEnum):
    """What to do when a cell cannot fully consume its allocated budget."""

    # Report the deficit and continue (default).
    REPORT = "report"
    # Treat any unfulfilled allocation as a hard error for the run.
    FAIL = "fail"


class FamilyParameters(BaseModel):
    """Per-family scheduling parameters (not the numerical attack parameters)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    weight: float = Field(gt=0.0)
    # Optional per-family overrides of the global window-length bounds.
    min_window: int | None = Field(default=None, ge=1)
    max_window: int | None = Field(default=None, ge=1)
    # Hours of prior observations required in the same segment before a window start.
    # Replay and context-aware stealth need history; spike/bias/drift default to none.
    min_history_hours: int = Field(default=0, ge=0)
    # Number of selected targets; must match the attack config under either budget policy.
    # A window of L positions reserves L * variable_count cell units, or L row units.
    variable_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_window_bounds(self) -> FamilyParameters:
        if (
            self.min_window is not None
            and self.max_window is not None
            and self.min_window > self.max_window
        ):
            raise ValueError("min_window must not exceed max_window")
        return self


class SchedulingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    global_seed: int = Field(ge=0)
    # Fraction of finite target observations to target, in [0, 1].
    attack_budget: float = Field(ge=0.0, le=1.0)
    # Partitions that make up the budget scope (denominator and placement).
    budget_scope: tuple[str, ...] = Field(min_length=1)
    enabled_families: tuple[AttackFamily, ...] = Field(min_length=1)
    family_parameters: dict[AttackFamily, FamilyParameters]
    min_window_length: int = Field(ge=1)
    max_window_length: int = Field(ge=1)
    max_attempts: int = Field(gt=0)
    # Overlap of reserved windows for the same station. False for the MVP.
    allow_overlap: bool = False
    insufficient_policy: InsufficientPolicy = InsufficientPolicy.REPORT
    # Fractional quota denominator remains finite observations of eligibility_target.
    # The unit declares whether that quota buys rows or individual variable cells.
    budget_unit: Literal["cells", "rows"] = "cells"
    max_modified_rows: int | None = Field(default=None, ge=0)
    max_modified_cells: int | None = Field(default=None, ge=0)
    eligibility_target: Literal["SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC"] = "SOIL_MOISTURE_5"

    @model_validator(mode="after")
    def _check_consistency(self) -> SchedulingConfig:
        if len(set(self.enabled_families)) != len(self.enabled_families):
            raise ValueError("enabled_families contains duplicates")
        if len(set(self.budget_scope)) != len(self.budget_scope):
            raise ValueError("budget_scope contains duplicate partitions")
        if self.min_window_length > self.max_window_length:
            raise ValueError("min_window_length must not exceed max_window_length")
        if self.allow_overlap:
            raise ValueError("allow_overlap=true is not supported in this phase")

        missing = [
            family for family in self.enabled_families if family not in self.family_parameters
        ]
        if missing:
            raise ValueError(f"missing family_parameters for enabled families: {missing}")
        extra = [family for family in self.family_parameters if family not in self.enabled_families]
        if extra:
            raise ValueError(f"family_parameters given for disabled families: {extra}")

        for family in self.enabled_families:
            params = self.family_parameters[family]
            low = params.min_window if params.min_window is not None else self.min_window_length
            high = params.max_window if params.max_window is not None else self.max_window_length
            if low > high:
                raise ValueError(f"family {family.value}: effective window bounds are inverted")
        return self

    def window_bounds(self, family: AttackFamily) -> tuple[int, int]:
        """Return the effective (min, max) window length for ``family``."""
        params = self.family_parameters[family]
        low = params.min_window if params.min_window is not None else self.min_window_length
        high = params.max_window if params.max_window is not None else self.max_window_length
        return low, high


def load_scheduling_config(path: Path) -> SchedulingConfig:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"scheduling configuration not found: {path}") from None
    except OSError as exc:
        raise ConfigError(f"cannot read scheduling configuration {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"scheduling configuration {path} must contain a mapping")
    try:
        return SchedulingConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid scheduling configuration {path}:\n{exc}") from exc
