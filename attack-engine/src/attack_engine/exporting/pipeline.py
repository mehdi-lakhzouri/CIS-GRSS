"""Top-level Phase 5 orchestration: build, validate, then atomically write one export run.

This module is the single entry point CLI wires into: it never re-executes an
attack, only arranges and writes the results of an already-committed
:class:`~attack_engine.attacks.executor.ExecutionSummary`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq

from attack_engine.attacks.config import AttackExecutionConfig
from attack_engine.attacks.executor import ExecutionSummary
from attack_engine.dataset.loader import LoadedDataset
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.exporting import builder, contracts
from attack_engine.exporting import provenance as provenance_module
from attack_engine.exporting.handover import build_handover_markdown
from attack_engine.exporting.manifest import build_attack_manifest
from attack_engine.exporting.models import (
    ExportConfiguration,
    ExportError,
    ExportResult,
)
from attack_engine.exporting.validation import require_valid, run_consistency_checks
from attack_engine.exporting.writer import (
    hash_files,
    run_in_temp_directory,
    write_csv,
    write_json,
    write_parquet,
    write_sha256sums,
    write_text,
)
from attack_engine.scheduling.scheduler import ScheduleResult
from attack_engine.utils.hashing import sha256_file

ATTACKED_FILE = "uscrn_attacked.parquet"
DETECTOR_INPUT_FILE = "detector_input.parquet"
GROUND_TRUTH_FILE = "ground_truth.parquet"
MANIFEST_FILE = "attack_manifest.csv"
CONTRACT_FILE = "data_contract.json"
PROVENANCE_FILE = "run_provenance.json"
HANDOVER_FILE = "HANDOVER.md"
# Files hashed into run_provenance.json; HANDOVER.md, the provenance file itself
# and SHA256SUMS.txt are covered only by SHA256SUMS.txt (see module docstring).
_PROVENANCE_HASHED_FILES = (
    ATTACKED_FILE,
    DETECTOR_INPUT_FILE,
    GROUND_TRUTH_FILE,
    MANIFEST_FILE,
    CONTRACT_FILE,
)


def default_run_id(global_seed: int, generated_at: datetime) -> str:
    return f"run-{generated_at.strftime('%Y%m%dT%H%M%SZ')}-seed{global_seed}-{uuid4().hex[:6]}"


def export_run(
    *,
    dataset: LoadedDataset,
    schema: DatasetSchema,
    partitions: PartitionPlan,
    scheduling_config_path: Path,
    attack_config: AttackExecutionConfig,
    attack_params_path: Path,
    plan: ScheduleResult,
    summary: ExecutionSummary,
    config: ExportConfiguration,
) -> ExportResult:
    frame = dataset.frame
    _verify_unchanged(Path(dataset.metadata.path), dataset.metadata.sha256, "input dataset")
    calibration_sha_before: str | None = None
    if attack_config.stealth is not None:
        calibration_path = attack_config.stealth.calibration_path
        calibration_sha_before = sha256_file(calibration_path)

    attacked = builder.build_attacked_target(frame, schema, summary)
    partition_labels = builder.assign_row_partitions(frame, schema, partitions)
    context_columns = contracts.resolve_detector_context_columns(schema, config)

    audit = builder.build_audit_dataset(frame, schema, partitions, attacked, partition_labels)
    ground_truth = builder.build_ground_truth(frame, schema, attacked, partition_labels)
    detector_input = builder.build_detector_input(
        frame, schema, attacked, partition_labels, context_columns
    )
    manifest = build_attack_manifest(plan, summary)
    contract = contracts.build_dataset_contract(schema, partitions, config, context_columns)

    checks = run_consistency_checks(
        frame, schema, summary, audit, ground_truth, detector_input, manifest
    )
    require_valid(checks)

    generated_at = datetime.now(UTC)
    run_id = config.run_id or default_run_id(attack_config.global_seed, generated_at)

    def write_all(run_dir: Path) -> None:
        write_parquet(run_dir / ATTACKED_FILE, audit)
        write_parquet(run_dir / DETECTOR_INPUT_FILE, detector_input)
        write_parquet(run_dir / GROUND_TRUTH_FILE, ground_truth)
        write_csv(run_dir / MANIFEST_FILE, manifest)
        write_json(run_dir / CONTRACT_FILE, contract.model_dump(mode="json"))

        hashed = hash_files(run_dir, list(_PROVENANCE_HASHED_FILES))
        run_provenance = provenance_module.build_run_provenance(
            run_id=run_id,
            dataset=dataset,
            schema=schema,
            partitions=partitions,
            scheduling_config_path=scheduling_config_path,
            attack_config=attack_config,
            attack_params_path=attack_params_path,
            plan=plan,
            summary=summary,
            validation_passed=all(check.passed for check in checks),
            validation_messages=tuple(
                f"{check.code}: {check.message}" for check in checks if not check.passed
            ),
            output_files=hashed,
        )
        write_json(run_dir / PROVENANCE_FILE, run_provenance.model_dump(mode="json"))
        write_text(
            run_dir / HANDOVER_FILE,
            build_handover_markdown(contract, run_provenance, len(frame)),
        )

        all_files = [*_PROVENANCE_HASHED_FILES, PROVENANCE_FILE, HANDOVER_FILE]
        write_sha256sums(run_dir, hash_files(run_dir, all_files))

    run_dir = run_in_temp_directory(config.output_root, run_id, write_all)

    file_sha256 = hash_files(run_dir, [*_PROVENANCE_HASHED_FILES, PROVENANCE_FILE, HANDOVER_FILE])
    _read_back_verify(run_dir, file_sha256, len(frame), len(manifest))

    _verify_unchanged(Path(dataset.metadata.path), dataset.metadata.sha256, "input dataset")
    if attack_config.stealth is not None:
        _verify_unchanged(
            attack_config.stealth.calibration_path, calibration_sha_before, "calibration artifact"
        )

    return ExportResult(
        run_id=run_id,
        run_dir=run_dir,
        row_count=len(frame),
        finite_target_count=int(ground_truth["is_evaluable"].sum()),
        missing_target_count=int((~ground_truth["is_evaluable"]).sum()),
        modified_count=int(ground_truth["is_attacked"].sum()),
        per_family_committed={
            family.value: execution.committed for family, execution in summary.per_family.items()
        },
        manifest_row_count=len(manifest),
        file_sha256=file_sha256,
        validation_passed=all(check.passed for check in checks),
        validation_messages=tuple(
            f"{check.code}: {check.message}" for check in checks if not check.passed
        ),
        generated_at=generated_at,
    )


def _verify_unchanged(path: Path, expected_sha256: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ExportError(
            f"{label} changed during export: expected SHA-256 {expected_sha256}, got {actual} "
            f"for {path}"
        )


def _read_back_verify(
    run_dir: Path, file_sha256: dict[str, str], expected_rows: int, expected_manifest_rows: int
) -> None:
    """Independently reopen every written file and check its actual contents."""
    for name, expected_hash in file_sha256.items():
        path = run_dir / name
        if not path.is_file():
            raise ExportError(f"expected output file is missing after export: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ExportError(f"read-back SHA-256 mismatch for {path}")

    for name in (ATTACKED_FILE, DETECTOR_INPUT_FILE, GROUND_TRUTH_FILE):
        table = pq.read_table(run_dir / name)
        if table.num_rows != expected_rows:
            raise ExportError(
                f"{name} has {table.num_rows} rows after read-back, expected {expected_rows}"
            )
    manifest = pd.read_csv(run_dir / MANIFEST_FILE)
    if len(manifest) != expected_manifest_rows:
        raise ExportError(
            f"{MANIFEST_FILE} has {len(manifest)} rows after read-back, "
            f"expected {expected_manifest_rows}"
        )
    for name in (CONTRACT_FILE, PROVENANCE_FILE):
        json.loads((run_dir / name).read_text(encoding="utf-8"))
