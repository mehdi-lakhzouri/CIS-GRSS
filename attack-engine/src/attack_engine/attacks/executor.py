"""In-memory orchestration of attack execution over a scheduling plan.

The executor reads the clean DataFrame, runs each reserved window's transformation,
commits the number of observations actually modified to the budget ledger, releases
unused or failed reservations, and returns a structured in-memory summary. It never
mutates the clean DataFrame and never writes any file; dataset export and manifests
belong to a later phase.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np
import pandas as pd

from attack_engine.attacks import base, multitarget
from attack_engine.attacks.calibration import load_calibration
from attack_engine.attacks.config import (
    EXECUTABLE_FAMILIES,
    AttackConfigError,
    AttackExecutionConfig,
)
from attack_engine.attacks.context import MultiStealthContextProvider, StealthContextProvider
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackResult,
    AttackSettings,
    CellKey,
    ReplayHistory,
    TargetBounds,
)
from attack_engine.attacks.multistealth import fit_secondary_calibration
from attack_engine.attacks.registry import transform_for
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.dataset.temporal import to_utc
from attack_engine.scheduling import identifiers
from attack_engine.scheduling.eligibility import EligibilityEngine, Segment
from attack_engine.scheduling.models import AttackFamily, AttackWindow, ScopeCell
from attack_engine.scheduling.scheduler import ScheduleResult


class ExecutionError(Exception):
    """A hard inconsistency during execution (e.g. overlapping writes)."""


class UnsupportedFamilyPolicy(StrEnum):
    """What to do when a plan contains a family this phase cannot execute."""

    # Release the reservation and report the skip explicitly (never silent).
    SKIP = "skip"
    # Fail the whole run rather than execute a partial plan.
    ERROR = "error"


@dataclass(frozen=True)
class FamilyExecution:
    family: AttackFamily
    planned_windows: int
    applied_windows: int
    rejected_windows: int
    reserved: int
    # Structural upper bound, before data-dependent rejection, clipping or quantization.
    # Spike is capped at max_points per window; the drift ramp starts at zero elapsed time.
    max_modifiable: int
    committed: int
    skipped_windows: int = 0
    allocated: int = 0
    modified_rows: int = 0
    modified_cells: int = 0
    per_variable_modifications: dict[str, int] = field(default_factory=dict)

    @property
    def unreserved(self) -> int:
        """Quota for which the Phase 2 planner could not reserve window capacity."""
        return self.allocated - self.reserved

    @property
    def unfulfilled_quota(self) -> int:
        """Modification quota not achieved, including unreserved and unused capacity."""
        return self.allocated - self.committed

    @property
    def deficit(self) -> int:
        """Reserved capacity not turned into modifications (data- and family-limited)."""
        return self.reserved - self.committed

    @property
    def structural_deficit(self) -> int:
        """Reserved capacity that is unreachable regardless of data (reserved - max_modifiable)."""
        return self.reserved - self.max_modifiable


@dataclass(frozen=True)
class ExecutionSummary:
    seed: int
    requested_quota: int
    reserved_before: int
    committed_total: int
    released_total: int
    per_family: dict[AttackFamily, FamilyExecution]
    results: tuple[AttackResult, ...]
    frame_unchanged: bool
    skipped_families: tuple[AttackFamily, ...] = field(default_factory=tuple)
    skipped_window_count: int = 0
    # Budget is counted in modified cells (one per (row, variable)). With a single target
    # variable every modification is one cell on a distinct row, so committed_cells equals
    # committed_total and committed_rows equals it too. The three stay distinct once a single
    # event can modify several variables of the same row.
    committed_cells: int = 0
    committed_rows: int = 0
    budget_unit: str = "cells"
    per_variable_modifications: dict[str, int] = field(default_factory=dict)
    secondary_calibration: dict | None = None

    @property
    def deficit_vs_quota(self) -> int:
        return self.requested_quota - self.committed_total

    @property
    def max_modifiable_total(self) -> int:
        return sum(execution.max_modifiable for execution in self.per_family.values())

    @property
    def applied_results(self) -> tuple[AttackResult, ...]:
        return tuple(result for result in self.results if result.applied)

    @property
    def rejected_results(self) -> tuple[AttackResult, ...]:
        return tuple(result for result in self.results if not result.applied)

    @property
    def context_total(self) -> int:
        return sum(int(result.diagnostics.extra.get("context_total", 0)) for result in self.results)

    @property
    def context_available(self) -> int:
        return sum(
            int(result.diagnostics.extra.get("context_available", 0)) for result in self.results
        )

    @property
    def calibration_sha256(self) -> str | None:
        return next(
            (
                str(result.diagnostics.extra["calibration_sha256"])
                for result in self.results
                if "calibration_sha256" in result.diagnostics.extra
            ),
            None,
        )


@dataclass(frozen=True)
class AttackExecutor:
    schema: DatasetSchema
    partitions: PartitionPlan
    config: AttackExecutionConfig
    unsupported_policy: UnsupportedFamilyPolicy = UnsupportedFamilyPolicy.SKIP

    def execute(self, frame: pd.DataFrame, plan: ScheduleResult) -> ExecutionSummary:
        self._require_parameters(plan)
        self._check_unsupported(plan)
        settings = AttackSettings(
            bounds=TargetBounds(self.config.target_low, self.config.target_high),
            quantum=self.config.quantum,
            variable_constraints=self.config.variable_constraints,
            relationships=self.config.relationships,
        )
        configured_columns = {
            column
            for family in plan.families
            if self.config.has_parameters(family)
            for column in self.config.target_columns_for(family)
        }
        missing = configured_columns - set(frame.columns)
        if missing:
            raise ExecutionError(f"configured target columns missing from frame: {sorted(missing)}")
        self._validate_variable_counts(plan)
        histories = self._build_histories(frame, plan)
        context_provider = self._context_provider(frame, plan)
        secondary_provider = self._secondary_provider(frame, plan)
        times = to_utc(frame[self.schema.timestamps.column])
        checksum_before = _frame_checksum(frame)

        reserved_before = plan.ledger.total_reserved
        results: list[AttackResult] = []
        written_cells: set[CellKey] = set()
        written_rows: set[int] = set()
        skipped: Counter[AttackFamily] = Counter()
        emitted: dict[int, float] = {}
        emitted_cells: dict[CellKey, float] = {}
        # Adjacent observations only: do not bridge station, partition or temporal gaps.
        previous, following = self._neighbors(frame, times)

        # Causal execution order, including other families' earlier emissions. Return
        # results in original plan order below to preserve the existing public interface.
        for window in sorted(
            plan.windows, key=lambda w: (w.station, w.first_observation_time, w.window_id)
        ):
            family = window.family
            if family not in EXECUTABLE_FAMILIES:
                # Policy already enforced by _check_unsupported; SKIP releases the reservation.
                plan.ledger.cancel(window.window_id)
                skipped[family] += 1
                continue

            labels = list(window.row_labels)
            target_columns = self.config.target_columns_for(family)
            clean_values = tuple(float(v) for v in frame.loc[labels, target_columns[0]].to_numpy())
            extra_clean = {
                column: tuple(float(v) for v in frame.loc[labels, column].to_numpy())
                for column in target_columns[1:]
                if column in frame
            }
            is_stealth = family is AttackFamily.STEALTH
            request = AttackRequest(
                window=window,
                clean_values=clean_values,
                elapsed_hours=tuple(
                    (time - times.loc[window.row_labels[0]]).total_seconds() / 3600
                    for time in times.loc[labels]
                ),
                history=histories.get(window.cell) if family is AttackFamily.REPLAY else None,
                stealth_context=context_provider.for_window(window, emitted)
                if is_stealth and context_provider is not None
                else None,
                multistealth_context=secondary_provider.secondary_context(window)
                if is_stealth and secondary_provider is not None
                else None,
                target_columns=target_columns,
                extra_clean=extra_clean,
                reference_clean={
                    column: tuple(frame.loc[labels, column].to_numpy(dtype=float))
                    for column in multitarget.VARIABLE_BOUNDS
                    if column in frame
                },
                previous_values={
                    column: emitted_cells.get(
                        (previous[labels[0]], column), float(frame.at[previous[labels[0]], column])
                    )
                    for column in target_columns
                    if labels[0] in previous
                },
                next_values={
                    column: float(frame.at[following[labels[-1]], column])
                    for column in target_columns
                    if labels[-1] in following
                },
                previous_offsets={
                    column: emitted_cells.get(
                        (previous[labels[0]], column), float(frame.at[previous[labels[0]], column])
                    )
                    - float(frame.at[previous[labels[0]], column])
                    for column in target_columns
                    if labels[0] in previous
                },
                boundary_interval_hours=self.schema.timestamps.interval.total_seconds() / 3600,
            )
            rng = base_generator(self.config.global_seed, family, window.window_id)
            result = transform_for(family)(
                request, self.config.parameters_for(family), rng, settings
            )
            result = multitarget.validate_result(request, result, settings)
            if result.applied and (
                (
                    plan.max_modified_rows is not None
                    and len(written_rows | result.modified_row_labels()) > plan.max_modified_rows
                )
                or (
                    plan.max_modified_cells is not None
                    and len(written_cells) + result.modified_cells > plan.max_modified_cells
                )
            ):
                result = base.rejected_result(
                    request,
                    family,
                    AttackRejectionReason.BUDGET_LIMIT,
                    "coordinated event would exceed configured row/cell cap; atomic rejection",
                )
            results.append(result)

            if result.applied and result.modified_cells > 0:
                committed = (
                    result.modified_cells if plan.budget_unit == "cells" else result.modified_rows
                )
                reservation = plan.ledger.reservation(window.window_id)
                if committed > reservation.reserved_count:
                    raise ExecutionError(f"{window.window_id}: result exceeds reserved budget")
                self._record_written(result, written_cells)
                plan.ledger.commit(window.window_id, committed)
                written_rows.update(result.modified_row_labels())
                for column in result.columns:
                    for label, value, changed in zip(
                        column.row_labels, column.candidate_values, column.changed_mask, strict=True
                    ):
                        if changed:
                            emitted_cells[(label, column.column)] = value
                            if column.column == self.schema.target_column:
                                emitted[label] = value
            else:
                plan.ledger.cancel(window.window_id)

        frame_unchanged = _frame_checksum(frame) == checksum_before
        result_by_id = {result.window_id: result for result in results}
        results = [result_by_id[w.window_id] for w in plan.windows if w.window_id in result_by_id]
        summary = self._summarize(plan, results, reserved_before, skipped, frame_unchanged)
        if secondary_provider is not None:
            summary = replace(
                summary,
                secondary_calibration=secondary_provider.secondary_calibration.model_dump(
                    mode="json"
                ),
            )
        return summary

    def _neighbors(
        self, frame: pd.DataFrame, times: pd.Series
    ) -> tuple[dict[int, int], dict[int, int]]:
        from attack_engine.dataset.temporal import assign_partitions

        interval = self.schema.timestamps.interval
        partition = assign_partitions(times, self.partitions, interval)
        ordered = frame.assign(_time=times, _partition=partition).sort_values(
            [self.schema.station_column, "_time"]
        )
        contiguous = (
            ordered._time.diff().eq(interval)
            & ordered[self.schema.station_column].eq(ordered[self.schema.station_column].shift())
            & ordered._partition.eq(ordered._partition.shift())
        )
        labels = ordered.index.to_numpy()
        right = labels[contiguous.to_numpy()]
        left = np.roll(labels, 1)[contiguous.to_numpy()]
        return dict(zip(right, left, strict=True)), dict(zip(left, right, strict=True))

    # Setup helpers -----------------------------------------------------------------

    def _context_provider(
        self, frame: pd.DataFrame, plan: ScheduleResult
    ) -> StealthContextProvider | None:
        if not plan.windows_for(AttackFamily.STEALTH) or self.config.stealth is None:
            return None
        calibration = load_calibration(self.config.stealth.calibration_path)
        calibration.require_compatible(self.config.stealth, self.schema, self.partitions)
        return StealthContextProvider(frame, self.schema, self.partitions, plan, calibration)

    def _secondary_provider(
        self, frame: pd.DataFrame, plan: ScheduleResult
    ) -> MultiStealthContextProvider | None:
        params = self.config.stealth
        if (
            not plan.windows_for(AttackFamily.STEALTH)
            or params is None
            or not params.secondary_columns
        ):
            return None
        calibration = load_calibration(params.calibration_path)
        calibration.require_compatible(params, self.schema, self.partitions)
        secondary = fit_secondary_calibration(frame, self.schema, self.partitions, params)
        return MultiStealthContextProvider(
            frame,
            self.schema,
            self.partitions,
            plan,
            calibration,
            params.secondary_columns,
            secondary,
        )

    def _validate_variable_counts(self, plan: ScheduleResult) -> None:
        """The scheduling variable_count (reserved cells) must match the attack target count."""
        for family in {window.family for window in plan.windows} & set(EXECUTABLE_FAMILIES):
            windows = plan.windows_for(family)
            reserved = sum(
                ledger.reserved for ledger in plan.ledger.cell_ledgers() if ledger.family is family
            )
            variables = (
                len(self.config.target_columns_for(family)) if plan.budget_unit == "cells" else 1
            )
            expected = sum(w.length for w in windows) * variables
            if reserved != expected:
                raise ExecutionError(
                    f"{family.value}: scheduling reserved {reserved} cells but the attack config "
                    f"targets {len(self.config.target_columns_for(family))} variable(s) "
                    f"(expected {expected}); set family_parameters.{family.value}.variable_count "
                    "to the number of target_columns"
                )
            target_count = len(self.config.target_columns_for(family))
            if family in plan.variable_counts and plan.variable_counts[family] != target_count:
                raise ExecutionError(
                    f"{family.value}: variable_count does not match target_columns"
                )
            for window in windows:
                reservation = plan.ledger.reservation(window.window_id)
                if reservation.reserved_count != window.length * variables:
                    raise ExecutionError(f"{window.window_id}: inconsistent per-event reservation")

    def _require_parameters(self, plan: ScheduleResult) -> None:
        planned = {window.family for window in plan.windows}
        for family in planned & set(EXECUTABLE_FAMILIES):
            if not self.config.has_parameters(family):
                raise AttackConfigError(
                    f"plan contains {family.value!r} windows but no parameters are configured"
                )

    def _check_unsupported(self, plan: ScheduleResult) -> None:
        unsupported = sorted(
            {window.family for window in plan.windows} - set(EXECUTABLE_FAMILIES),
            key=lambda family: family.value,
        )
        if unsupported and self.unsupported_policy is UnsupportedFamilyPolicy.ERROR:
            names = ", ".join(family.value for family in unsupported)
            raise ExecutionError(
                f"plan contains non-executable families ({names}); this phase executes only "
                f"{', '.join(f.value for f in EXECUTABLE_FAMILIES)}. "
                "Set the unsupported-family policy to 'skip' to release their reservations."
            )

    def _max_modifiable(self, family: AttackFamily, windows: tuple[AttackWindow, ...]) -> int:
        """Largest achievable modified-cell count: primary positions times every variable."""
        if family not in EXECUTABLE_FAMILIES or not self.config.has_parameters(family):
            return 0
        params = self.config.parameters_for(family)
        variables = len(self.config.target_columns_for(family))
        if family is AttackFamily.SPIKE:
            positions = sum(min(window.length, params.max_points) for window in windows)
        elif family is AttackFamily.DRIFT:
            # elapsed_hours starts at zero, so the ramp's first point is always held.
            positions = sum(max(0, window.length - 1) for window in windows)
        elif family is AttackFamily.STEALTH:
            positions = sum(w.length for w in windows if w.partition in params.attack_partitions)
        else:
            positions = sum(window.length for window in windows)
        return positions * variables

    def _build_histories(
        self, frame: pd.DataFrame, plan: ScheduleResult
    ) -> dict[ScopeCell, ReplayHistory]:
        replay_cells = {
            window.cell for window in plan.windows if window.family is AttackFamily.REPLAY
        }
        if not replay_cells:
            return {}
        targets = self.config.target_columns_for(AttackFamily.REPLAY)
        target, secondary = targets[0], targets[1:]
        replay_schema = self.schema.model_copy(update={"target_column": target})
        report = EligibilityEngine(replay_schema, self.partitions, plan.scope).analyze(frame)
        histories: dict[ScopeCell, ReplayHistory] = {}
        for cell in replay_cells:
            eligibility = report.cell(cell)
            segments: tuple[Segment, ...] = eligibility.segments if eligibility else ()

            def column_runs(column: str, segs: tuple[Segment, ...] = segments) -> tuple:
                return tuple(
                    tuple(float(v) for v in frame.loc[list(segment.row_labels), column].to_numpy())
                    for segment in segs
                )

            histories[cell] = ReplayHistory(
                run_times=tuple(segment.times for segment in segments),
                run_values=column_runs(target),
                run_labels=tuple(segment.row_labels for segment in segments),
                run_secondary={column: column_runs(column) for column in secondary},
            )
        return histories

    @staticmethod
    def _record_written(result: AttackResult, written: set[CellKey]) -> None:
        """Guard against two committed events writing the same (station, time, variable) cell.

        Keyed on cells rather than rows so that, once multi-variable events exist, attacks on
        different variables of the same row do not falsely collide while a genuine double-write
        of one variable still raises.
        """
        keys = result.modified_cell_keys()
        collision = written.intersection(keys)
        if collision or len(keys) != len(set(keys)):
            label, column = next(iter(collision or keys))
            raise ExecutionError(
                f"cell (row {label}, {column}) would be modified by more than one window "
                f"(at {result.window_id})"
            )
        written.update(keys)

    def _summarize(
        self,
        plan: ScheduleResult,
        results: list[AttackResult],
        reserved_before: int,
        skipped: Counter[AttackFamily],
        frame_unchanged: bool,
    ) -> ExecutionSummary:
        per_family: dict[AttackFamily, FamilyExecution] = {}
        for family in plan.families:
            family_results = [result for result in results if result.family is family]
            applied = [result for result in family_results if result.applied]
            windows = plan.windows_for(family)
            variables = (
                len(self.config.target_columns_for(family))
                if self.config.has_parameters(family)
                else 1
            )
            factor = variables if plan.budget_unit == "cells" else 1
            reserved = sum(window.length for window in windows) * factor
            per_family[family] = FamilyExecution(
                family=family,
                planned_windows=len(windows),
                applied_windows=len(applied),
                rejected_windows=len(family_results) - len(applied),
                reserved=reserved,
                max_modifiable=self._max_modifiable(family, windows)
                // (variables if plan.budget_unit == "rows" else 1),
                committed=sum(
                    result.modified_cells if plan.budget_unit == "cells" else result.modified_rows
                    for result in applied
                ),
                skipped_windows=skipped.get(family, 0),
                allocated=plan.family_quotas[family],
                modified_rows=len(
                    {label for result in applied for label in result.modified_row_labels()}
                ),
                modified_cells=sum(result.modified_cells for result in applied),
                per_variable_modifications=_variable_counts(family_results),
            )
        committed = plan.ledger.total_committed
        applied = [result for result in results if result.applied]
        committed_cells = sum(result.modified_cells for result in applied)
        committed_rows = len(
            {label for result in applied for label in result.modified_row_labels()}
        )
        return ExecutionSummary(
            seed=self.config.global_seed,
            requested_quota=plan.requested_quota,
            reserved_before=reserved_before,
            committed_total=committed,
            released_total=reserved_before - committed,
            per_family=per_family,
            results=tuple(results),
            frame_unchanged=frame_unchanged,
            skipped_families=tuple(sorted(skipped, key=lambda family: family.value)),
            skipped_window_count=int(sum(skipped.values())),
            committed_cells=committed_cells,
            committed_rows=committed_rows,
            budget_unit=plan.budget_unit,
            per_variable_modifications=_variable_counts(results),
        )


def base_generator(seed: int, family: AttackFamily, window_id: str) -> np.random.Generator:
    """Deterministic per-window generator for execution draws."""
    return identifiers.derive_generator(seed, "execute", family.value, window_id)


def _frame_checksum(frame: pd.DataFrame) -> str:
    """Order-sensitive checksum of the whole frame, to verify it is never mutated."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    digest.update(str(tuple(zip(frame.columns, map(str, frame.dtypes), strict=True))).encode())
    return digest.hexdigest()


def _variable_counts(results: list[AttackResult]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(result.per_variable_modifications)
    return dict(sorted(counts.items()))
