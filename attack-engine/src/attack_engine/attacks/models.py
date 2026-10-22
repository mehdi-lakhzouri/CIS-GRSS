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

from attack_engine.scheduling.models import AttackFamily, AttackWindow


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

    @property
    def length(self) -> int:
        return len(self.clean_values)


@dataclass(frozen=True)
class StealthContext:
    """Observable context and strictly earlier emissions, never future clean target lags."""

    times: tuple[datetime, ...]
    measurements: dict[str, tuple[float, ...]]
    history_times: tuple[datetime, ...]
    history_values: tuple[float, ...]
    calibration: ContextCalibration


@dataclass(frozen=True)
class ReplayHistory:
    """Clean, continuous finite runs of the same cell, available as replay sources.

    Each run is a maximal hourly-contiguous block of finite target observations in
    the window's station and partition, read from the immutable clean dataset.
    """

    run_times: tuple[tuple[datetime, ...], ...]
    run_values: tuple[tuple[float, ...], ...]
    run_labels: tuple[tuple[int, ...], ...]


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
class AttackResult:
    """Outcome of one transformation: applied with a changed mask, or rejected."""

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

    @property
    def modified_count(self) -> int:
        return self.diagnostics.modified_count

    @property
    def applied(self) -> bool:
        return self.status is AttackStatus.APPLIED
