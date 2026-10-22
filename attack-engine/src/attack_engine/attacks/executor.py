"""In-memory orchestration of attack execution over a scheduling plan.

The executor reads the clean DataFrame, runs each reserved window's transformation,
commits the number of observations actually modified to the budget ledger, releases
unused or failed reservations, and returns a structured in-memory summary. It never
mutates the clean DataFrame and never writes any file; dataset export and manifests
belong to a later phase.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from attack_engine.attacks.calibration import load_calibration
from attack_engine.attacks.config import (
    EXECUTABLE_FAMILIES,
    AttackConfigError,
    AttackExecutionConfig,
)
from attack_engine.attacks.context import StealthContextProvider
from attack_engine.attacks.models import (
    AttackRequest,
    AttackResult,
    AttackSettings,
    ReplayHistory,
    TargetBounds,
)
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
        )
        target = self.schema.target_column
        histories = self._build_histories(frame, plan)
        context_provider = self._context_provider(frame, plan)
        times = to_utc(frame[self.schema.timestamps.column])
        checksum_before = _frame_checksum(frame)

        reserved_before = plan.ledger.total_reserved
        results: list[AttackResult] = []
        written_labels: set[int] = set()
        skipped: Counter[AttackFamily] = Counter()
        emitted: dict[int, float] = {}

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

            clean_values = tuple(
                float(value) for value in frame.loc[list(window.row_labels), target].to_numpy()
            )
            request = AttackRequest(
                window=window,
                clean_values=clean_values,
                elapsed_hours=tuple(
                    (time - times.loc[window.row_labels[0]]).total_seconds() / 3600
                    for time in times.loc[list(window.row_labels)]
                ),
                history=histories.get(window.cell) if family is AttackFamily.REPLAY else None,
                stealth_context=context_provider.for_window(window, emitted)
                if context_provider is not None and family is AttackFamily.STEALTH
                else None,
            )
            rng = base_generator(self.config.global_seed, family, window.window_id)
            result = transform_for(family)(
                request, self.config.parameters_for(family), rng, settings
            )
            results.append(result)

            if result.applied and result.modified_count > 0:
                self._record_written(window, result, written_labels)
                plan.ledger.commit(window.window_id, result.modified_count)
                emitted.update(
                    {
                        label: value
                        for label, value, changed in zip(
                            result.row_labels,
                            result.candidate_values,
                            result.changed_mask,
                            strict=True,
                        )
                        if changed
                    }
                )
            else:
                plan.ledger.cancel(window.window_id)

        frame_unchanged = _frame_checksum(frame) == checksum_before
        result_by_id = {result.window_id: result for result in results}
        results = [result_by_id[w.window_id] for w in plan.windows if w.window_id in result_by_id]
        return self._summarize(plan, results, reserved_before, skipped, frame_unchanged)

    # Setup helpers -----------------------------------------------------------------

    def _context_provider(
        self, frame: pd.DataFrame, plan: ScheduleResult
    ) -> StealthContextProvider | None:
        if not plan.windows_for(AttackFamily.STEALTH) or self.config.stealth is None:
            return None
        calibration = load_calibration(self.config.stealth.calibration_path)
        calibration.require_compatible(self.config.stealth, self.schema, self.partitions)
        return StealthContextProvider(frame, self.schema, self.partitions, plan, calibration)

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
        """Largest achievable modification count for a family given its windows."""
        if family not in EXECUTABLE_FAMILIES or not self.config.has_parameters(family):
            return 0
        params = self.config.parameters_for(family)
        if family is AttackFamily.SPIKE:
            return sum(min(window.length, params.max_points) for window in windows)
        if family is AttackFamily.DRIFT:
            # elapsed_hours starts at zero, so the ramp's first point is always held,
            # including the legacy hold_first=False configuration.
            return sum(max(0, window.length - 1) for window in windows)
        if family is AttackFamily.STEALTH:
            return sum(w.length for w in windows if w.partition in params.attack_partitions)
        return sum(window.length for window in windows)

    def _build_histories(
        self, frame: pd.DataFrame, plan: ScheduleResult
    ) -> dict[ScopeCell, ReplayHistory]:
        replay_cells = {
            window.cell for window in plan.windows if window.family is AttackFamily.REPLAY
        }
        if not replay_cells:
            return {}
        report = EligibilityEngine(self.schema, self.partitions, plan.scope).analyze(frame)
        target = self.schema.target_column
        histories: dict[ScopeCell, ReplayHistory] = {}
        for cell in replay_cells:
            eligibility = report.cell(cell)
            segments: tuple[Segment, ...] = eligibility.segments if eligibility else ()
            histories[cell] = ReplayHistory(
                run_times=tuple(segment.times for segment in segments),
                run_values=tuple(
                    tuple(
                        float(value)
                        for value in frame.loc[list(segment.row_labels), target].to_numpy()
                    )
                    for segment in segments
                ),
                run_labels=tuple(segment.row_labels for segment in segments),
            )
        return histories

    @staticmethod
    def _record_written(window, result: AttackResult, written: set[int]) -> None:
        for label, changed in zip(result.row_labels, result.changed_mask, strict=True):
            if changed:
                if label in written:
                    raise ExecutionError(
                        f"row {label} would be modified by more than one window "
                        f"(at {window.window_id})"
                    )
                written.add(label)

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
            reserved = sum(window.length for window in windows)
            per_family[family] = FamilyExecution(
                family=family,
                planned_windows=len(windows),
                applied_windows=len(applied),
                rejected_windows=len(family_results) - len(applied),
                reserved=reserved,
                max_modifiable=self._max_modifiable(family, windows),
                committed=sum(result.modified_count for result in applied),
                skipped_windows=skipped.get(family, 0),
                allocated=plan.family_quotas[family],
            )
        committed = plan.ledger.total_committed
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
        )


def base_generator(seed: int, family: AttackFamily, window_id: str) -> np.random.Generator:
    """Deterministic per-window generator for execution draws."""
    return identifiers.derive_generator(seed, "execute", family.value, window_id)


def _frame_checksum(frame: pd.DataFrame) -> int:
    """Order-sensitive checksum of the whole frame, to verify it is never mutated."""
    return int(pd.util.hash_pandas_object(frame, index=True).to_numpy().sum())
