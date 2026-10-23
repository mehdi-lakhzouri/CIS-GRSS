"""The attack-window scheduler: deterministic planning over eligibility and budget.

The scheduler draws non-overlapping reservation windows, consuming budget quotas,
and produces a :class:`ScheduleResult`. It is independent of the attack
transformations: it reserves *where* an attack could go, never computes or writes
a modified value, and never mutates the source DataFrame.

Determinism and ordering
------------------------
Cells (station, partition) are processed in a fixed sorted order and, within each
cell, families in their configured order. Reserved windows must not overlap for
the same station (``allow_overlap=false``); that constraint is shared across
families within a cell, so an earlier family claims capacity first. Each
(family, station, partition) stream draws from its own generator derived from the
global seed and stable names (see :mod:`attack_engine.scheduling.identifiers`), so
adding or reordering families or stations does not shift another stream's draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

import pandas as pd

from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.scheduling import identifiers
from attack_engine.scheduling.budget import (
    BudgetLedger,
    allocate_largest_remainder,
    budget_quota,
)
from attack_engine.scheduling.config import InsufficientPolicy, SchedulingConfig
from attack_engine.scheduling.eligibility import EligibilityEngine, Segment
from attack_engine.scheduling.models import (
    AttackFamily,
    AttackWindow,
    RejectionReason,
    ScopeCell,
    WindowRejection,
)


class SchedulingError(Exception):
    """Scheduling could not satisfy a hard requirement (e.g. strict deficit policy)."""


@dataclass(frozen=True)
class ScheduleResult:
    seed: int
    scope: tuple[str, ...]
    families: tuple[AttackFamily, ...]
    requested_budget: float
    finite_denominator: int
    requested_quota: int
    family_quotas: dict[AttackFamily, int]
    denominators: dict[ScopeCell, int]
    windows: tuple[AttackWindow, ...]
    rejections: tuple[WindowRejection, ...]
    ledger: BudgetLedger
    budget_unit: Literal["cells", "rows"] = "cells"
    max_modified_rows: int | None = None
    max_modified_cells: int | None = None
    eligibility_target: str = "SOIL_MOISTURE_5"
    variable_counts: dict[AttackFamily, int] = field(default_factory=dict)

    @property
    def reserved_total(self) -> int:
        return self.ledger.total_reserved

    @property
    def unfulfilled_total(self) -> int:
        return self.ledger.total_allocated - self.ledger.total_reserved

    def windows_for(self, family: AttackFamily) -> tuple[AttackWindow, ...]:
        return tuple(window for window in self.windows if window.family is family)


class _FreeGaps:
    """Free start-offset gaps per segment within one cell, shared across families.

    Each gap ``(lo, hi)`` means offsets ``lo..hi`` (inclusive) are unreserved and
    may host the *start* of a window as long as the whole window fits. Reserving a
    window splits the gap, so windows for different families in the same cell can
    never overlap.
    """

    def __init__(self, segments: tuple[Segment, ...]) -> None:
        self._gaps: list[list[tuple[int, int]]] = [
            [(0, segment.length - 1)] for segment in segments
        ]
        self._segments = segments

    def start_ranges(self, length: int, min_history: int) -> list[tuple[int, int, int, int]]:
        """Valid start ranges as ``(segment_index, gap_index, start_lo, start_hi)``.

        ``start_hi`` is inclusive. Only ranges that can host a full window of
        ``length`` while leaving ``min_history`` observations before the start are
        returned, in segment then start order.
        """
        ranges: list[tuple[int, int, int, int]] = []
        for segment_index, gaps in enumerate(self._gaps):
            for gap_index, (lo, hi) in enumerate(gaps):
                start_lo = max(lo, min_history)
                start_hi = hi - (length - 1)
                if start_lo <= start_hi:
                    ranges.append((segment_index, gap_index, start_lo, start_hi))
        return ranges

    def reserve(self, segment_index: int, gap_index: int, start: int, length: int) -> None:
        lo, hi = self._gaps[segment_index][gap_index]
        replacement: list[tuple[int, int]] = []
        if lo <= start - 1:
            replacement.append((lo, start - 1))
        if start + length <= hi:
            replacement.append((start + length, hi))
        self._gaps[segment_index][gap_index : gap_index + 1] = replacement


@dataclass(frozen=True)
class AttackWindowScheduler:
    schema: DatasetSchema
    partitions: PartitionPlan
    config: SchedulingConfig

    def plan(self, frame: pd.DataFrame) -> ScheduleResult:
        eligibility_schema = self.schema.model_copy(
            update={"target_column": self.config.eligibility_target}
        )
        report = EligibilityEngine(
            eligibility_schema, self.partitions, self.config.budget_scope
        ).analyze(frame)
        denominators = dict(report.finite_by_cell)
        family_quotas, allocations = self._allocate(denominators)

        ledger = BudgetLedger(allocations, denominators)
        windows: list[AttackWindow] = []
        rejections: list[WindowRejection] = list(report.rejections)

        for cell in self._sorted_cells(denominators):
            eligibility = report.cell(cell)
            if eligibility is None or not eligibility.segments:
                continue
            free = _FreeGaps(eligibility.segments)
            for family in self.config.enabled_families:
                self._plan_cell(
                    family, cell, eligibility.segments, free, ledger, windows, rejections
                )

        self._apply_deficit_policy(ledger, rejections)
        total_finite = sum(denominators.values())
        return ScheduleResult(
            seed=self.config.global_seed,
            scope=self.config.budget_scope,
            families=self.config.enabled_families,
            requested_budget=self.config.attack_budget,
            finite_denominator=total_finite,
            requested_quota=budget_quota(self.config.attack_budget, total_finite),
            family_quotas=family_quotas,
            denominators=denominators,
            windows=tuple(windows),
            rejections=tuple(rejections),
            ledger=ledger,
            budget_unit=self.config.budget_unit,
            max_modified_rows=self.config.max_modified_rows,
            max_modified_cells=self.config.max_modified_cells,
            eligibility_target=self.config.eligibility_target,
            variable_counts={
                family: params.variable_count
                for family, params in self.config.family_parameters.items()
            },
        )

    # Budget allocation -------------------------------------------------------------

    def _allocate(
        self, denominators: dict[ScopeCell, int]
    ) -> tuple[dict[AttackFamily, int], dict[tuple[AttackFamily, ScopeCell], int]]:
        total_finite = sum(denominators.values())
        total_quota = budget_quota(self.config.attack_budget, total_finite)

        weights = {
            family: self.config.family_parameters[family].weight
            for family in self.config.enabled_families
        }
        family_quotas_raw = allocate_largest_remainder(total_quota, weights)
        family_quotas = {
            family: family_quotas_raw[family] for family in self.config.enabled_families
        }

        allocations: dict[tuple[AttackFamily, ScopeCell], int] = {}
        for family in self.config.enabled_families:
            cell_quota = allocate_largest_remainder(family_quotas[family], denominators)
            for cell in denominators:
                allocations[(family, cell)] = cell_quota[cell]
        return family_quotas, allocations

    @staticmethod
    def _sorted_cells(denominators: dict[ScopeCell, int]) -> list[ScopeCell]:
        return sorted(denominators, key=lambda cell: (cell.partition, cell.station))

    # Per-cell window drawing -------------------------------------------------------

    def _plan_cell(
        self,
        family: AttackFamily,
        cell: ScopeCell,
        segments: tuple[Segment, ...],
        free: _FreeGaps,
        ledger: BudgetLedger,
        windows: list[AttackWindow],
        rejections: list[WindowRejection],
    ) -> None:
        if ledger.allocated(family, cell) == 0:
            return
        min_window, max_window = self.config.window_bounds(family)
        family_params = self.config.family_parameters[family]
        min_history = family_params.min_history_hours
        # Budget is counted in cells: a window of L positions costs L * variable_count cells.
        variables = family_params.variable_count if self.config.budget_unit == "cells" else 1
        interval = self.schema.timestamps.interval
        generator = identifiers.derive_generator(
            self.config.global_seed, family.value, cell.station, cell.partition
        )

        attempts = 0
        no_fit = 0
        while attempts < self.config.max_attempts:
            # How many window positions the remaining cell budget can still afford.
            affordable = ledger.remaining(family, cell) // variables
            if affordable < min_window:
                break
            attempts += 1
            length = int(generator.integers(min_window, min(max_window, affordable) + 1))
            ranges = free.start_ranges(length, min_history)
            if not ranges:
                no_fit += 1
                continue
            segment_index, gap_index, start = self._draw_start(generator, ranges)
            window = self._build_window(
                family, cell, segments[segment_index], start, length, interval, len(windows)
            )
            ledger.reserve(window.window_id, family, cell, length * variables)
            free.reserve(segment_index, gap_index, start, length)
            windows.append(window)

        self._record_unfulfilled(
            family, cell, ledger, attempts, no_fit, min_window * variables, rejections
        )

    @staticmethod
    def _draw_start(generator, ranges: list[tuple[int, int, int, int]]) -> tuple[int, int, int]:
        """Uniformly pick one concrete start across all valid start positions."""
        sizes = [start_hi - start_lo + 1 for _, _, start_lo, start_hi in ranges]
        total = sum(sizes)
        pick = int(generator.integers(total))
        cumulative = 0
        for (segment_index, gap_index, start_lo, _), size in zip(ranges, sizes, strict=True):
            if pick < cumulative + size:
                return segment_index, gap_index, start_lo + (pick - cumulative)
            cumulative += size
        raise AssertionError("draw index out of range")  # pragma: no cover

    def _build_window(
        self,
        family: AttackFamily,
        cell: ScopeCell,
        segment: Segment,
        start: int,
        length: int,
        interval: timedelta,
        sequence: int,
    ) -> AttackWindow:
        row_labels = segment.row_labels[start : start + length]
        times = segment.times[start : start + length]
        window_id = f"{family.value}-{cell.station}-{cell.partition}-{sequence:04d}"
        return AttackWindow(
            window_id=window_id,
            family=family,
            station=cell.station,
            partition=cell.partition,
            length=length,
            first_observation_time=times[0],
            last_observation_time=times[-1],
            end_exclusive=times[-1] + interval,
            row_labels=row_labels,
        )

    def _record_unfulfilled(
        self,
        family: AttackFamily,
        cell: ScopeCell,
        ledger: BudgetLedger,
        attempts: int,
        no_fit: int,
        min_window: int,
        rejections: list[WindowRejection],
    ) -> None:
        remaining = ledger.remaining(family, cell)
        if remaining <= 0:
            return
        if attempts >= self.config.max_attempts:
            reason = RejectionReason.MAX_ATTEMPTS_REACHED
        elif remaining < min_window:
            reason = RejectionReason.REMAINDER_BELOW_MIN_WINDOW
        elif no_fit > 0:
            reason = RejectionReason.NO_CANDIDATE_START
        else:
            reason = RejectionReason.SEGMENT_TOO_SHORT
        detail = (
            f"{remaining} of {ledger.allocated(family, cell)} units unplaced after "
            f"{attempts} attempts ({no_fit} with no fitting window)"
        )
        rejections.append(
            WindowRejection(
                family=family,
                station=cell.station,
                partition=cell.partition,
                reason=reason,
                detail=detail,
                count=remaining,
            )
        )

    def _apply_deficit_policy(
        self, ledger: BudgetLedger, rejections: list[WindowRejection]
    ) -> None:
        if self.config.insufficient_policy is not InsufficientPolicy.FAIL:
            return
        deficit = ledger.total_allocated - ledger.total_reserved
        if deficit > 0:
            raise SchedulingError(
                f"insufficient_policy=fail: {deficit} budget units could not be reserved"
            )
