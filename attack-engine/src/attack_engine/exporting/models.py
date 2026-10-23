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
        if self.run_id is not None and (
            self.run_id in {".", ".."} or any(c in self.run_id for c in "/\\:")
        ):
            raise ValueError("run_id must be a single directory name")
        if len(set(self.detector_label_partitions)) != len(self.detector_label_partitions) or set(
            self.detector_label_partitions
        ) - {"train", "validation"}:
            raise ValueError("detector labels may contain only unique train/validation partitions")
        if self.detector_context_columns is not None and len(
            set(self.detector_context_columns)
        ) != len(self.detector_context_columns):
            raise ValueError("detector_context_columns must be unique")
        return self


class DatasetContract(BaseModel):
    """Explicit, versioned schema of every artifact produced by one export run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0", "2.0"] = EXPORT_SCHEMA_VERSION
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
    # Multi-target foundation (defaults describe the current single-target run): the target
    # variables an event may attack, the clean-target -> attacked-column name map, and the
    # columns of the optional variable-level (cell) ground-truth table.
    target_columns: tuple[str, ...] = ()
    attacked_columns: dict[str, str] = Field(default_factory=dict)
    variable_ground_truth_columns: tuple[str, ...] = ()
    row_evaluability: str = "finite primary target"
    cell_ground_truth_coverage: str = (
        "every source row and selected variable, including negatives and missing values"
    )
    detector_release_directory: str = "detector"
    test_ground_truth_policy: str = (
        "evaluator only until predictions are frozen; never in detector release"
    )
    target_units: dict[str, str] = Field(default_factory=dict)
    artifact_audiences: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_multitarget(self) -> DatasetContract:
        # Keep the contract self-consistent without changing single-target construction sites.
        if not self.target_columns:
            object.__setattr__(self, "target_columns", (self.target_column,))
        if not self.attacked_columns:
            object.__setattr__(
                self, "attacked_columns", {self.target_column: self.attacked_target_column}
            )
        return self


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


class GroundTruthCellRecord(BaseModel):
    """One (station, time, variable) cell of the variable-level ground truth.

    The exporter writes every selected cell, including negatives and missing values.
    The legacy foundation projection helper remains available for changed-only views.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    station: str
    time: AwareDatetime
    partition: str | None
    column: str
    clean_value: float | None
    attacked_value: float | None
    is_attacked: bool
    attack_type: str | None
    attack_id: str | None
    is_evaluable: bool


class MultiVariableGroundTruthRecord(GroundTruthRecord):
    """Row union with aligned per-target values and exact changed-variable membership."""

    clean_values: list[float | None]
    attacked_values: list[float | None]
    changed_columns: list[str]
    evaluable_columns: list[str]
    modified_cells: int = Field(ge=0)
    attack_types: list[str]
    attack_ids: list[str]


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
    quota_deficit: int = 0
    modified_rows: int = 0
    modified_cells: int = 0
    per_variable_modifications: dict[str, int] = Field(default_factory=dict)


class RunProvenance(BaseModel):
    """Everything needed to reproduce one export run; never hashes itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    export_schema_version: Literal["1.0", "2.0"] = EXPORT_SCHEMA_VERSION
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
    target_columns: tuple[str, ...] = ()
    budget_unit: str = "cells"
    max_modified_rows: int | None = None
    max_modified_cells: int | None = None
    modified_rows: int = 0
    modified_cells: int = 0
    per_variable_modifications: dict[str, int] = Field(default_factory=dict)
    per_partition_modifications: dict[str, dict[str, int]] = Field(default_factory=dict)
    train_preserved: bool = False
    resolved_attack_configuration: dict = Field(default_factory=dict)
    secondary_calibration: dict | None = None
    secondary_calibration_sha256: str | None = None
    engine_source_sha256: dict[str, str] = Field(default_factory=dict)

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
    modified_rows: int = 0
    modified_cells: int = 0
    per_variable_modifications: dict[str, int] = Field(default_factory=dict)
    target_columns: tuple[str, ...] = ()
    cell_ground_truth_row_count: int = 0
    detector_release_dir: Path | None = None
    budget_unit: str = "cells"
    quota_deficit: int = 0
    evaluable_row_count: int = 0
    evaluable_cell_count: int = 0
