"""Typed, immutable objects exchanged by the scheduling components.

A reservation uses the half-open index interval ``[start, end_exclusive)``:
for a window of ``length`` consecutive hourly observations,
``end_exclusive = last_observation_time + interval``. The physically measured
span of the window runs from ``first_observation_time - interval`` to
``last_observation_time``; the two conventions are kept distinct.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AttackFamily(StrEnum):
    SPIKE = "spike"
    BIAS = "bias"
    DRIFT = "drift"
    REPLAY = "replay"
    STEALTH = "stealth"


class RejectionReason(StrEnum):
    NO_ELIGIBLE_SEGMENT = "no_eligible_segment"
    SEGMENT_TOO_SHORT = "segment_too_short"
    INSUFFICIENT_HISTORY = "insufficient_history"
    MISSING_TARGET = "missing_target_in_window"
    ALREADY_RESERVED = "window_already_reserved"
    BUDGET_EXHAUSTED = "cell_budget_exhausted"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"
    NO_CANDIDATE_START = "no_candidate_start"
    # Remaining allocation is smaller than the family's minimum window length, so no
    # further window can be placed without truncating below the configured minimum.
    REMAINDER_BELOW_MIN_WINDOW = "remainder_below_min_window"


class ScopeCell(BaseModel):
    """One (station, partition) cell: the unit of budget accounting."""

    model_config = ConfigDict(frozen=True)

    station: str
    partition: str

    def label(self) -> str:
        return f"{self.station}/{self.partition}"


class AttackWindow(BaseModel):
    """A reserved, contiguous block of hourly observations for one station and family.

    ``row_labels`` are the original DataFrame index labels of the observations, so
    Phase 3 can locate the rows without the scheduler reordering anything.
    """

    model_config = ConfigDict(frozen=True)

    window_id: str
    family: AttackFamily
    station: str
    partition: str
    length: int = Field(ge=1)
    first_observation_time: datetime
    last_observation_time: datetime
    # Half-open reservation bound: last_observation_time + interval.
    end_exclusive: datetime
    row_labels: tuple[int, ...]

    @model_validator(mode="after")
    def _check_shape(self) -> AttackWindow:
        if len(self.row_labels) != self.length:
            raise ValueError("row_labels length must equal the window length")
        if self.first_observation_time > self.last_observation_time:
            raise ValueError("first_observation_time must not exceed last_observation_time")
        if self.end_exclusive <= self.last_observation_time:
            raise ValueError("end_exclusive must follow last_observation_time")
        return self

    @property
    def cell(self) -> ScopeCell:
        return ScopeCell(station=self.station, partition=self.partition)


class WindowRejection(BaseModel):
    """A traceable reason a candidate could not become a reserved window."""

    model_config = ConfigDict(frozen=True)

    # None for cell-level reasons that are not tied to a particular family.
    family: AttackFamily | None = None
    station: str
    partition: str
    reason: RejectionReason
    detail: str = ""
    count: int = 1
