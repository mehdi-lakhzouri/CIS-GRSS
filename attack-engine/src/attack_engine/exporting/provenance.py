"""Builds the ``run_provenance.json`` record.

Everything needed to reproduce a run is captured here: input and configuration
file identities, the global seed, dependency versions, partition boundaries,
budget accounting and per-family statistics. The provenance document never
contains its own hash (see :mod:`attack_engine.exporting.writer` for the
checksums file that covers it instead).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

from attack_engine.attacks.config import AttackExecutionConfig
from attack_engine.attacks.executor import ExecutionSummary
from attack_engine.dataset.loader import LoadedDataset
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.exporting.models import FamilyProvenance, PartitionBoundary, RunProvenance
from attack_engine.scheduling.scheduler import ScheduleResult
from attack_engine.utils.hashing import sha256_file

_TRACKED_DEPENDENCIES = ("numpy", "pandas", "pyarrow", "pydantic", "pyyaml")


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_DEPENDENCIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def build_run_provenance(
    *,
    run_id: str,
    dataset: LoadedDataset,
    schema: DatasetSchema,
    partitions: PartitionPlan,
    scheduling_config_path: Path,
    attack_config: AttackExecutionConfig,
    attack_params_path: Path,
    plan: ScheduleResult,
    summary: ExecutionSummary,
    validation_passed: bool,
    validation_messages: tuple[str, ...],
    output_files: dict[str, str],
) -> RunProvenance:
    calibration_path = attack_config.stealth.calibration_path if attack_config.stealth else None
    per_family = tuple(
        FamilyProvenance(
            family=family.value,
            planned_windows=execution.planned_windows,
            applied_windows=execution.applied_windows,
            rejected_windows=execution.rejected_windows,
            allocated=execution.allocated,
            reserved=execution.reserved,
            committed=execution.committed,
            deficit=execution.deficit,
        )
        for family, execution in summary.per_family.items()
    )
    return RunProvenance(
        run_id=run_id,
        generated_at=datetime.now(UTC),
        input_path=dataset.metadata.path,
        input_sha256=dataset.metadata.sha256,
        dataset_schema_name=schema.name,
        calibration_path=str(calibration_path) if calibration_path else None,
        calibration_file_sha256=sha256_file(calibration_path) if calibration_path else None,
        calibration_logical_sha256=summary.calibration_sha256,
        scheduling_config_path=str(Path(scheduling_config_path).resolve()),
        scheduling_config_sha256=sha256_file(scheduling_config_path),
        attack_params_path=str(Path(attack_params_path).resolve()),
        attack_params_sha256=sha256_file(attack_params_path),
        global_seed=attack_config.global_seed,
        python_version=sys.version,
        dependency_versions=dependency_versions(),
        partitions=tuple(
            PartitionBoundary(name=p.name, start=p.start, end=p.end) for p in partitions.partitions
        ),
        total_rows=dataset.metadata.row_count,
        finite_target_denominator=plan.finite_denominator,
        requested_budget_fraction=plan.requested_budget,
        requested_quota=plan.requested_quota,
        allocated_budget=plan.ledger.total_allocated,
        reserved_capacity=summary.reserved_before,
        committed_modifications=summary.committed_total,
        unfulfilled_budget=plan.ledger.total_allocated - summary.committed_total,
        per_family=per_family,
        output_files=output_files,
        validation_passed=validation_passed,
        validation_messages=validation_messages,
    )
