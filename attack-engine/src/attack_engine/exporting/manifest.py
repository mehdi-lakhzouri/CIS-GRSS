"""Event-level attack manifest: one row per planned window, win or lose.

Built directly from the scheduler's plan (every reserved window) and the
executor's results (keyed by window id), so a rejected or skipped window is
represented with zero modifications rather than silently omitted.
"""

from __future__ import annotations

import json

import pandas as pd

from attack_engine.attacks.executor import ExecutionSummary
from attack_engine.attacks.models import AttackResult, AttackStatus
from attack_engine.exporting.contracts import MANIFEST_COLUMNS, MULTI_MANIFEST_COLUMNS
from attack_engine.exporting.models import AttackManifestRecord, EventStatus
from attack_engine.scheduling.models import AttackFamily, AttackWindow
from attack_engine.scheduling.scheduler import ScheduleResult


def build_attack_manifest(
    plan: ScheduleResult,
    summary: ExecutionSummary,
    *,
    multi: bool = False,
    family_targets: dict[AttackFamily, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    results_by_window: dict[str, AttackResult] = {
        result.window_id: result for result in summary.results
    }
    skipped_families: set[AttackFamily] = set(summary.skipped_families)
    records = [
        _manifest_record(window, results_by_window.get(window.window_id), skipped_families)
        for window in plan.windows
    ]
    manifest = pd.DataFrame.from_records(
        [record.model_dump(mode="json") for record in records], columns=MANIFEST_COLUMNS
    )
    if multi:
        extra = []
        for window in plan.windows:
            result = results_by_window.get(window.window_id)
            targets = (family_targets or {}).get(
                window.family, result.resolved_target_columns if result else ()
            )
            reservation = plan.ledger.reservation(window.window_id)
            extra.append(
                {
                    "target_columns": json.dumps(targets),
                    "changed_variables": json.dumps(result.changed_variables if result else ()),
                    "modified_rows": result.modified_rows if result else 0,
                    "modified_cells": result.modified_cells if result else 0,
                    "per_variable_modifications": json.dumps(
                        result.per_variable_modifications if result else dict.fromkeys(targets, 0),
                        sort_keys=True,
                    ),
                    "budget_unit": plan.budget_unit,
                    "reserved_budget": reservation.reserved_count,
                    "committed_budget": reservation.committed_count or 0,
                    "reserved_cells": window.length * len(targets),
                }
            )
        manifest = pd.concat(
            [
                manifest,
                pd.DataFrame(extra, columns=MULTI_MANIFEST_COLUMNS[len(MANIFEST_COLUMNS) :]),
            ],
            axis=1,
        )
        manifest["requested_capacity"] = manifest["reserved_budget"]
    return manifest


def _manifest_record(
    window: AttackWindow, result: AttackResult | None, skipped_families: set[AttackFamily]
) -> AttackManifestRecord:
    if result is None:
        status = EventStatus.SKIPPED if window.family in skipped_families else EventStatus.REJECTED
        return AttackManifestRecord(
            attack_id=window.window_id,
            family=window.family.value,
            station=window.station,
            partition=window.partition,
            window_start=window.first_observation_time,
            window_end_exclusive=window.end_exclusive,
            length=window.length,
            status=status,
            requested_capacity=window.length,
            reserved_positions=window.length,
            modified_count=0,
            rejection_reason=None,
            requested_magnitude=None,
            sign=None,
            mean_abs_delta=None,
            max_abs_delta=None,
            saturated_count=None,
        )
    status = EventStatus.APPLIED if result.status is AttackStatus.APPLIED else EventStatus.REJECTED
    diagnostics = result.diagnostics
    replay_source = result.replay_source
    return AttackManifestRecord(
        attack_id=result.window_id,
        family=result.family.value,
        station=result.station,
        partition=result.partition,
        window_start=window.first_observation_time,
        window_end_exclusive=window.end_exclusive,
        length=window.length,
        status=status,
        requested_capacity=window.length,
        reserved_positions=len(result.row_labels),
        modified_count=result.modified_count,
        rejection_reason=result.rejection_reason.value if result.rejection_reason else None,
        requested_magnitude=diagnostics.requested_magnitude,
        sign=diagnostics.sign,
        mean_abs_delta=diagnostics.mean_abs_delta,
        max_abs_delta=diagnostics.max_abs_delta,
        saturated_count=diagnostics.saturated_count,
        replay_source_first_time=replay_source.first_time if replay_source else None,
        replay_source_last_time=replay_source.last_time if replay_source else None,
        replay_source_age_hours=replay_source.age_hours if replay_source else None,
    )
