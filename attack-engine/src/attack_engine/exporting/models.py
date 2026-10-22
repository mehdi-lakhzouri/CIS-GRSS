"""Typed, immutable export data contracts (Phase 5).

These Pydantic models describe the *shape* and *configuration* of one export
run; they hold no DataFrame and perform no I/O. Construction of the actual
datasets lives in :mod:`attack_engine.exporting.builder`, orchestration and
file writing in :mod:`attack_engine.exporting.pipeline` and
:mod:`attack_engine.exporting.writer`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

EXPORT_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class ExportError(Exception):
    """The export could not be produced or written safely."""


class EventStatus(StrEnum):
    """Execution status of one planned attack window, for the manifest."""

    APPLIED = "applied"
    REJECTED = "rejected"
    # Planned for a family this phase does not execute; reservation released.
    SKIPPED = "skipped"
    # Reserved but released without ever being attempted (not produced today,
    # kept so the manifest's status model stays stable if that becomes possible).
    CANCELLED = "cancelled"


class ExportConfiguration(BaseModel):
    """User-facing export options. Everything else is derived from upstream results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_root: Path
    # None generates a timestamped, seed-qualified identifier.
    run_id: str | None = None
    # None uses the project default allowlist (see attack_engine.exporting.contracts).
    detector_context_columns: tuple[str, ...] | None = None
    # Partitions whose ground-truth labels the AI team may access before freezing
    # predictions. Test labels stay restricted to the evaluator.
    detector_label_partitions: tuple[str, ...] = ("train", "validation")

    @model_validator(mode="after")
    def _check_run_id(self) -> ExportConfiguration:
        if self.run_id is not None and not self.run_id.strip():
            raise ValueError("run_id must not be blank")
        return self


class DatasetContract(BaseModel):
    """Explicit, versioned schema of every artifact produced by one export run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = EXPORT_SCHEMA_VERSION
    key_columns: tuple[str, ...]
    target_column: str
    attacked_target_column: str
    source_columns: tuple[str, ...]
    audit_extra_columns: tuple[str, ...]
    detector_input_columns: tuple[str, ...]
    detector_context_columns: tuple[str, ...]
    ground_truth_columns: tuple[str, ...]
    manifest_columns: tuple[str, ...]
    # Columns that must never reach the detector, with the reason they are privileged.
    excluded_from_detector: dict[str, str]
    partitions: tuple[str, ...]
    detector_label_partitions: tuple[str, ...]


class GroundTruthRecord(BaseModel):
    """One row of ``ground_truth.parquet``; documents the contract, not the writer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    station: str
    time: AwareDatetime
    partition: str | None
    clean_target: float | None
    attacked_target: float | None
    is_attacked: bool
    attack_type: str | None
    attack_id: str | None
    is_evaluable: bool


class AttackManifestRecord(BaseModel):
    """One row of ``attack_manifest.csv``: one planned attack-window event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attack_id: str
    family: str
    station: str
    partition: str
    window_start: AwareDatetime
    window_end_exclusive: AwareDatetime
    length: int = Field(ge=1)
    status: EventStatus
    requested_capacity: int = Field(ge=0)
    reserved_positions: int = Field(ge=0)
    modified_count: int = Field(ge=0)
    rejection_reason: str | None
    requested_magnitude: float | None
    sign: int | None
    mean_abs_delta: float | None
    max_abs_delta: float | None
    saturated_count: int | None
    replay_source_first_time: AwareDatetime | None = None
    replay_source_last_time: AwareDatetime | None = None
    replay_source_age_hours: float | None = None


class PartitionBoundary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    start: AwareDatetime
    end: AwareDatetime


class FamilyProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str
    planned_windows: int
    applied_windows: int
    rejected_windows: int
    allocated: int
    reserved: int
    committed: int
    deficit: int


class RunProvenance(BaseModel):
    """Everything needed to reproduce one export run; never hashes itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    export_schema_version: Literal["1.0"] = EXPORT_SCHEMA_VERSION
    run_id: str
    generated_at: AwareDatetime

    input_path: str
    input_sha256: str
    dataset_schema_name: str

    calibration_path: str | None
    calibration_file_sha256: str | None
    calibration_logical_sha256: str | None

    scheduling_config_path: str
    scheduling_config_sha256: str
    attack_params_path: str
    attack_params_sha256: str

    global_seed: int
    python_version: str
    dependency_versions: dict[str, str]

    partitions: tuple[PartitionBoundary, ...]
    total_rows: int
    finite_target_denominator: int

    requested_budget_fraction: float
    requested_quota: int
    allocated_budget: int
    reserved_capacity: int
    committed_modifications: int
    unfulfilled_budget: int

    per_family: tuple[FamilyProvenance, ...]

    # File name (relative to the run directory) -> SHA-256. Excludes this file
    # itself and SHA256SUMS.txt, which is written last and covers both.
    output_files: dict[str, str]

    validation_passed: bool
    validation_messages: tuple[str, ...]


class ExportRun(BaseModel):
    """Resolved identity of one export run, decided before any file is written."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    run_dir: Path
    created_at: AwareDatetime


class ExportResult(BaseModel):
    """Summary returned after a successful, read-back-verified export."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    run_dir: Path
    row_count: int
    finite_target_count: int
    missing_target_count: int
    modified_count: int
    per_family_committed: dict[str, int]
    manifest_row_count: int
    file_sha256: dict[str, str]
    validation_passed: bool
    validation_messages: tuple[str, ...]
    generated_at: datetime
