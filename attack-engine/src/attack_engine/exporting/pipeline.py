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

from attack_engine.attacks.calibration import load_calibration
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
CELL_GROUND_TRUTH_FILE = "ground_truth_cells.parquet"
DETECTOR_DIRECTORY = "detector"
ROW_LABELS_FILE = "row_labels.parquet"
CELL_LABELS_FILE = "cell_labels.parquet"
# Files hashed into run_provenance.json; HANDOVER.md, the provenance file itself
# and SHA256SUMS.txt are covered only by SHA256SUMS.txt (see module docstring).
_PROVENANCE_HASHED_FILES = (
    ATTACKED_FILE,
    DETECTOR_INPUT_FILE,
    GROUND_TRUTH_FILE,
    MANIFEST_FILE,
    CONTRACT_FILE,
    CELL_GROUND_TRUTH_FILE,
    *(
        f"{DETECTOR_DIRECTORY}/{name}"
        for name in (
            DETECTOR_INPUT_FILE,
            ROW_LABELS_FILE,
            CELL_LABELS_FILE,
            CONTRACT_FILE,
            HANDOVER_FILE,
            "SHA256SUMS.txt",
        )
    ),
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
    config_hashes = {
        Path(path): sha256_file(path) for path in (scheduling_config_path, attack_params_path)
    }
    _verify_unchanged(Path(dataset.metadata.path), dataset.metadata.sha256, "input dataset")
    _verify_source_frame(dataset)
    calibration_sha_before: str | None = None
    if attack_config.stealth is not None:
        calibration_path = attack_config.stealth.calibration_path
        calibration_sha_before = sha256_file(calibration_path)
        if summary.calibration_sha256 is not None and (
            load_calibration(calibration_path).calibration_sha256 != summary.calibration_sha256
        ):
            raise ExportError("calibration no longer matches the executed contextual reference")

    family_targets = {
        f: attack_config.target_columns_for(f)
        for f in plan.families
        if attack_config.has_parameters(f)
    }
    target_columns = tuple(
        dict.fromkeys(c for columns in family_targets.values() for c in columns)
    ) or (schema.target_column,)
    multi = target_columns != (schema.target_column,)
    attacked = builder.build_attacked_columns(frame, schema, summary, target_columns)
    partition_labels = builder.assign_row_partitions(frame, schema, partitions)
    context_columns = contracts.resolve_detector_context_columns(schema, config, target_columns)

    if multi:
        ground_truth = builder.build_multivariable_ground_truth(
            frame, schema, attacked, partition_labels
        )
        audit = builder.build_variable_audit_dataset(
            frame, attacked, ground_truth, partition_labels
        )
    else:
        primary = attacked[target_columns[0]]
        audit = builder.build_audit_dataset(frame, schema, partitions, primary, partition_labels)
        ground_truth = builder.build_ground_truth(frame, schema, primary, partition_labels)
    cell_ground_truth = builder.build_variable_ground_truth(
        frame, schema, attacked, partition_labels
    )
    detector_input = builder.build_variable_detector_input(
        frame, schema, attacked, partition_labels, context_columns
    )
    manifest = build_attack_manifest(plan, summary, multi=multi, family_targets=family_targets)
    contract = contracts.build_dataset_contract(
        schema, partitions, config, context_columns, target_columns
    )
    row_labels = builder.build_detector_labels(ground_truth, config.detector_label_partitions)
    cell_labels = builder.build_detector_labels(
        cell_ground_truth, config.detector_label_partitions, cells=True
    )

    checks = run_consistency_checks(
        frame,
        schema,
        summary,
        audit,
        ground_truth,
        detector_input,
        manifest,
        cell_ground_truth=cell_ground_truth,
        target_columns=target_columns,
        plan=plan,
        contract=contract,
    )
    require_valid(checks)

    generated_at = datetime.now(UTC)
    run_id = config.run_id or default_run_id(attack_config.global_seed, generated_at)

    def write_all(run_dir: Path) -> None:
        write_parquet(run_dir / ATTACKED_FILE, audit)
        write_parquet(run_dir / DETECTOR_INPUT_FILE, detector_input)
        write_parquet(run_dir / GROUND_TRUTH_FILE, ground_truth)
        write_parquet(run_dir / CELL_GROUND_TRUTH_FILE, cell_ground_truth)
        write_csv(run_dir / MANIFEST_FILE, manifest)
        write_json(run_dir / CONTRACT_FILE, contract.model_dump(mode="json"))
        release = run_dir / DETECTOR_DIRECTORY
        release.mkdir(mode=0o700)
        write_parquet(release / DETECTOR_INPUT_FILE, detector_input)
        write_parquet(release / ROW_LABELS_FILE, row_labels)
        write_parquet(release / CELL_LABELS_FILE, cell_labels)
        write_json(release / CONTRACT_FILE, contract.model_dump(mode="json"))
        from attack_engine.exporting.handover import build_detector_handover

        write_text(release / HANDOVER_FILE, build_detector_handover(contract))
        write_sha256sums(
            release,
            hash_files(
                release,
                [
                    DETECTOR_INPUT_FILE,
                    ROW_LABELS_FILE,
                    CELL_LABELS_FILE,
                    CONTRACT_FILE,
                    HANDOVER_FILE,
                ],
            ),
        )

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
            target_columns=target_columns,
            partition_statistics={
                name: {
                    "modified_rows": int(
                        ground_truth.loc[ground_truth.partition.eq(name), "is_attacked"].sum()
                    ),
                    "modified_cells": int(
                        cell_ground_truth.loc[
                            cell_ground_truth.partition.eq(name), "is_attacked"
                        ].sum()
                    ),
                }
                for name in contract.partitions
            },
            schema_version=contract.schema_version,
        )
        write_json(run_dir / PROVENANCE_FILE, run_provenance.model_dump(mode="json"))
        write_text(
            run_dir / HANDOVER_FILE,
            build_handover_markdown(contract, run_provenance, len(frame)),
        )

        all_files = [*_PROVENANCE_HASHED_FILES, PROVENANCE_FILE, HANDOVER_FILE]
        write_sha256sums(run_dir, hash_files(run_dir, all_files))
        # Read back and validate before publishing the directory, so failed checks never
        # leave a completed-looking export. Dataset values, identities and masks are rechecked.
        _read_back_verify(
            run_dir,
            hash_files(run_dir, all_files),
            len(frame),
            len(manifest),
            len(cell_ground_truth),
        )
        for path, expected in (
            (run_dir / CONTRACT_FILE, contract.model_dump(mode="json")),
            (release / CONTRACT_FILE, contract.model_dump(mode="json")),
            (run_dir / PROVENANCE_FILE, run_provenance.model_dump(mode="json")),
        ):
            if json.loads(path.read_text(encoding="utf-8")) != expected:
                raise ExportError(f"JSON metadata read-back mismatch: {path.name}")
        restored = {
            name: pd.read_parquet(run_dir / name)
            for name in (
                ATTACKED_FILE,
                DETECTOR_INPUT_FILE,
                GROUND_TRUTH_FILE,
                CELL_GROUND_TRUTH_FILE,
            )
        }
        restored_truth = restored[GROUND_TRUTH_FILE]
        for name in ("changed_columns", "evaluable_columns", "attack_types", "attack_ids"):
            if name in restored_truth:
                restored_truth[name] = restored_truth[name].map(list)
        require_valid(
            run_consistency_checks(
                frame,
                schema,
                summary,
                restored[ATTACKED_FILE],
                restored_truth,
                restored[DETECTOR_INPUT_FILE],
                pd.read_csv(run_dir / MANIFEST_FILE),
                cell_ground_truth=restored[CELL_GROUND_TRUTH_FILE],
                target_columns=target_columns,
                plan=plan,
                contract=contract,
            )
        )
        for filename, expected in (
            (ROW_LABELS_FILE, row_labels),
            (CELL_LABELS_FILE, cell_labels),
            (DETECTOR_INPUT_FILE, detector_input),
        ):
            actual = pd.read_parquet(release / filename)
            expected = expected.copy()
            # Parquet nulls may read back as None instead of pandas.NA when a label
            # column is entirely null (for example, an explicit zero-budget run).
            for label in ("attack_type", "attack_id"):
                if label in actual:
                    actual[label] = actual[label].astype("string")
                    expected[label] = expected[label].astype("string")
            for name in ("changed_columns", "attack_types", "attack_ids"):
                if name in actual:
                    actual[name] = actual[name].map(list)
            if (
                "partition" in actual
                and filename != DETECTOR_INPUT_FILE
                and actual.partition.eq("test").any()
            ):
                raise ExportError("test ground truth would enter the detector release")
            try:
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
            except AssertionError as exc:
                raise ExportError(f"detector release read-back mismatch: {filename}") from exc
        _verify_unchanged(Path(dataset.metadata.path), dataset.metadata.sha256, "input dataset")
        _verify_source_frame(dataset)
        for path, digest in config_hashes.items():
            _verify_unchanged(path, digest, "configuration")
        if attack_config.stealth is not None:
            _verify_unchanged(
                attack_config.stealth.calibration_path,
                calibration_sha_before,
                "calibration artifact",
            )

    run_dir = run_in_temp_directory(config.output_root, run_id, write_all)

    file_sha256 = hash_files(run_dir, [*_PROVENANCE_HASHED_FILES, PROVENANCE_FILE, HANDOVER_FILE])
    _read_back_verify(run_dir, file_sha256, len(frame), len(manifest), len(cell_ground_truth))

    _verify_unchanged(Path(dataset.metadata.path), dataset.metadata.sha256, "input dataset")
    if attack_config.stealth is not None:
        _verify_unchanged(
            attack_config.stealth.calibration_path, calibration_sha_before, "calibration artifact"
        )

    return ExportResult(
        run_id=run_id,
        run_dir=run_dir,
        row_count=len(frame),
        finite_target_count=int(frame[target_columns[0]].notna().sum()),
        missing_target_count=int(frame[target_columns[0]].isna().sum()),
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
        modified_rows=summary.committed_rows,
        modified_cells=summary.committed_cells,
        per_variable_modifications={
            c: summary.per_variable_modifications.get(c, 0) for c in target_columns
        },
        target_columns=target_columns,
        cell_ground_truth_row_count=len(cell_ground_truth),
        detector_release_dir=run_dir / DETECTOR_DIRECTORY,
        budget_unit=summary.budget_unit,
        quota_deficit=summary.deficit_vs_quota,
        evaluable_row_count=int(ground_truth.is_evaluable.sum()),
        evaluable_cell_count=int(cell_ground_truth.is_evaluable.sum()),
    )


def _verify_unchanged(path: Path, expected_sha256: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ExportError(
            f"{label} changed during export: expected SHA-256 {expected_sha256}, got {actual} "
            f"for {path}"
        )


def _verify_source_frame(dataset: LoadedDataset) -> None:
    with pq.ParquetFile(dataset.metadata.path) as source:
        original = source.read().to_pandas()
    try:
        pd.testing.assert_frame_equal(dataset.frame, original, check_exact=True)
    except AssertionError as exc:
        raise ExportError(
            "in-memory clean frame differs from the unchanged source dataset"
        ) from exc


def _read_back_verify(
    run_dir: Path,
    file_sha256: dict[str, str],
    expected_rows: int,
    expected_manifest_rows: int,
    expected_cells: int | None = None,
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
    if (
        expected_cells is not None
        and pq.read_table(run_dir / CELL_GROUND_TRUTH_FILE).num_rows != expected_cells
    ):
        raise ExportError("cell ground truth row count changed after read-back")
    recorded = {
        name: digest
        for digest, name in (
            line.split("  ", 1)
            for line in (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        )
    }
    if recorded != file_sha256:
        raise ExportError("SHA256SUMS.txt does not match the exported file inventory")
