"""Typed, immutable structures exchanged by attack transformations.

An :class:`AttackRequest` carries one clean window and the context a transformation
needs; an :class:`AttackResult` reports the candidate values, the mask of positions
that actually changed, the realized counts and diagnostics, or a traceable
rejection. No structure holds a mutable view of the source DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attack_engine.attacks.calibration import ContextCalibration
    from attack_engine.attacks.config import CrossVariableConstraint, VariableConstraints
    from attack_engine.attacks.multistealth import SecondaryCalibration

from attack_engine.scheduling.models import AttackFamily, AttackWindow

# Default channel for backward-compatible configurations; every family can select others.
DEFAULT_TARGET_COLUMN = "SOIL_MOISTURE_5"

# A modified cell is one (row, variable) pair. Because a row label uniquely identifies a
# (station, time) observation in the clean frame, a cell key is exactly a
# (station, time, variable) coordinate, which is what overlap detection operates on.
CellKey = tuple[int, str]


class SignPolicy(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    # A deterministic +1/-1 chosen from the window's generator.
    RANDOM = "random"


class AttackStatus(StrEnum):
    APPLIED = "applied"
    REJECTED = "rejected"


class AttackRejectionReason(StrEnum):
    MISSING_TARGET = "missing_target_in_window"
    WINDOW_TOO_SHORT = "window_too_short"
    INFEASIBLE_BOUNDS = "infeasible_within_bounds"
    SLOPE_TOO_STEEP = "slope_exceeds_maximum"
    NO_REPLAY_SOURCE = "no_eligible_replay_source"
    ZERO_EFFECT = "zero_effect_after_bounds_and_quantization"
    MISSING_CONTEXT = "missing_required_context"
    INSUFFICIENT_HISTORY = "insufficient_observable_history"
    NO_CALIBRATION = "no_station_calibration"
    FORBIDDEN_PARTITION = "stealth_partition_not_authorized"
    INVALID_CONTEXT = "invalid_context_timestamps"
    INFEASIBLE_CONTEXT = "empty_contextual_feasible_range"
    QUANTIZATION_INFEASIBLE = "no_feasible_quantized_value"
    BELOW_MIN_IMPACT = "below_minimum_useful_impact"
    TEMPORAL_CONSTRAINT = "temporal_constraint_violation"
    CROSS_VARIABLE_CONSTRAINT = "cross_variable_constraint_violation"
    BUDGET_LIMIT = "row_or_cell_budget_exceeded"


@dataclass(frozen=True)
class TargetBounds:
    """Inclusive physical bounds for the finite target channel."""

    low: float = 0.0
    high: float = 0.70

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("low bound must not exceed high bound")


@dataclass(frozen=True)
class AttackSettings:
    """Global numerical settings shared by every transformation."""

    bounds: TargetBounds = field(default_factory=TargetBounds)
    # Optional grid applied to candidate (modified) values only; never to clean values.
    quantum: float | None = None
    variable_constraints: dict[str, VariableConstraints] = field(default_factory=dict)
    relationships: tuple[CrossVariableConstraint, ...] = ()


@dataclass(frozen=True)
class ReplaySource:
    """A chosen historical source for a replay event (clean data only)."""

    first_time: datetime
    last_time: datetime
    age_hours: float
    row_labels: tuple[int, ...]
    candidate_count: int


@dataclass(frozen=True)
class AttackRequest:
    """Inputs to one transformation for one reserved window.

    ``clean_values`` are the clean target observations for the window rows, in time
    order. ``elapsed_hours`` is the elapsed time of each row relative to the first,
    used by time-based transformations. ``history`` is supplied only for families
    (replay) that read earlier clean observations of the same station and partition.
    ``stealth_context`` supplies current environmental measurements, a frozen train
    calibration, and strictly earlier emitted observations for the contextual family.
    """

    window: AttackWindow
    clean_values: tuple[float, ...]
    elapsed_hours: tuple[float, ...]
    history: ReplayHistory | None = None
    stealth_context: StealthContext | None = None
    # The target variable(s) this event may modify. Empty resolves to the single default
    # target. ``clean_values`` is always the primary (first) target's clean series; a
    # multi-variable request carries the secondary columns' clean series in ``extra_clean``.
    target_columns: tuple[str, ...] = ()
    extra_clean: dict[str, tuple[float, ...]] = field(default_factory=dict)
    # Supplied only for coordinated multi-variable stealth events.
    multistealth_context: MultiStealthContext | None = None
    # Clean reference channels include unselected variables needed by physical relationships
    # and the latent 5 cm reference for secondary-only contextual stealth.
    reference_clean: dict[str, tuple[float, ...]] = field(default_factory=dict)
    previous_values: dict[str, float] = field(default_factory=dict)
    next_values: dict[str, float] = field(default_factory=dict)
    previous_offsets: dict[str, float] = field(default_factory=dict)
    next_offsets: dict[str, float] = field(default_factory=dict)
    boundary_interval_hours: float = 1.0

    @property
    def length(self) -> int:
        return len(self.clean_values)

    @property
    def resolved_target_columns(self) -> tuple[str, ...]:
        return self.target_columns or (DEFAULT_TARGET_COLUMN,)

    @property
    def primary_target_column(self) -> str:
        return self.resolved_target_columns[0]


@dataclass(frozen=True)
class StealthContext:
    """Observable context and strictly earlier emissions, never future clean target lags."""

    times: tuple[datetime, ...]
    measurements: dict[str, tuple[float, ...]]
    history_times: tuple[datetime, ...]
    history_values: tuple[float, ...]
    calibration: ContextCalibration


@dataclass(frozen=True)
class MultiStealthContext:
    """Secondary-variable clean series and their train-fitted coupling for one window.

    The primary soil-moisture context lives in :pyattr:`AttackRequest.stealth_context`;
    this adds, per secondary column, the clean window values (so missing values are
    preserved and never fabricated) and the frozen coupling calibrated on train.
    """

    secondary_clean: dict[str, tuple[float, ...]]
    calibration: SecondaryCalibration


@dataclass(frozen=True)
class ReplayHistory:
    """Clean, continuous finite runs of the same cell, available as replay sources.

    Each run is a maximal hourly-contiguous block of finite target observations in
    the window's station and partition, read from the immutable clean dataset.
    """

    run_times: tuple[tuple[datetime, ...], ...]
    run_values: tuple[tuple[float, ...], ...]
    run_labels: tuple[tuple[int, ...], ...]
    # Per secondary column: clean source values aligned with each run (for multi-variable replay).
    run_secondary: dict[str, tuple[tuple[float, ...], ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class AttackDiagnostics:
    """Realized statistics of an attack event (zeros for a rejected event)."""

    requested_magnitude: float
    sign: int
    modified_count: int
    mean_abs_delta: float
    max_abs_delta: float
    saturated_count: int
    extra: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class ColumnModification:
    """One target variable's clean/candidate values and changed mask within an event."""

    column: str
    row_labels: tuple[int, ...]
    clean_values: tuple[float, ...]
    candidate_values: tuple[float, ...]
    changed_mask: tuple[bool, ...]

    @property
    def modified_count(self) -> int:
        return sum(self.changed_mask)

    @property
    def modified_labels(self) -> tuple[int, ...]:
        return tuple(
            label
            for label, changed in zip(self.row_labels, self.changed_mask, strict=True)
            if changed
        )

    def cell_keys(self) -> tuple[CellKey, ...]:
        return tuple((label, self.column) for label in self.modified_labels)


@dataclass(frozen=True)
class AttackResult:
    """Outcome of one transformation: applied with a changed mask, or rejected.

    The primary target's arrays (``clean_values``/``candidate_values``/``changed_mask``)
    are the backward-compatible single-target view. ``extra_columns`` carries any further
    selected variables that change. Use :pyattr:`columns` for the uniform per-variable view.
    """

    window_id: str
    family: AttackFamily
    station: str
    partition: str
    status: AttackStatus
    row_labels: tuple[int, ...]
    clean_values: tuple[float, ...]
    candidate_values: tuple[float, ...]
    changed_mask: tuple[bool, ...]
    diagnostics: AttackDiagnostics
    rejection_reason: AttackRejectionReason | None = None
    detail: str = ""
    replay_source: ReplaySource | None = None
    target_columns: tuple[str, ...] = ()
    extra_columns: tuple[ColumnModification, ...] = ()

    @property
    def modified_count(self) -> int:
        return self.diagnostics.modified_count

    @property
    def applied(self) -> bool:
        return self.status is AttackStatus.APPLIED

    @property
    def resolved_target_columns(self) -> tuple[str, ...]:
        return self.target_columns or (DEFAULT_TARGET_COLUMN,)

    @property
    def primary_target_column(self) -> str:
        return self.resolved_target_columns[0]

    @property
    def primary_column(self) -> ColumnModification:
        return ColumnModification(
            column=self.primary_target_column,
            row_labels=self.row_labels,
            clean_values=self.clean_values,
            candidate_values=self.candidate_values,
            changed_mask=self.changed_mask,
        )

    @property
    def columns(self) -> tuple[ColumnModification, ...]:
        """Uniform per-variable view: the primary target plus any extra columns."""
        return (self.primary_column, *self.extra_columns)

    @property
    def modified_cells(self) -> int:
        """Total modified (row, variable) cells across all target variables."""
        return sum(column.modified_count for column in self.columns)

    @property
    def modified_rows(self) -> int:
        """Distinct rows modified in at least one variable."""
        return len(self.modified_row_labels())

    def modified_row_labels(self) -> frozenset[int]:
        """Distinct row labels modified in at least one variable."""
        return frozenset(label for column in self.columns for label in column.modified_labels)

    def modified_cell_keys(self) -> tuple[CellKey, ...]:
        return tuple(key for column in self.columns for key in column.cell_keys())

    @property
    def per_variable_modifications(self) -> dict[str, int]:
        counts = dict.fromkeys(self.resolved_target_columns, 0)
        counts.update({column.column: column.modified_count for column in self.columns})
        return counts

    @property
    def changed_variables(self) -> tuple[str, ...]:
        return tuple(column for column, count in self.per_variable_modifications.items() if count)
