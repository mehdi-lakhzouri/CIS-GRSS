"""Construction of the three exported DataFrames from an already-committed execution.

Nothing here runs or recomputes an attack: it only reads
:class:`~attack_engine.attacks.executor.ExecutionSummary` results (each already
carrying its own changed mask and realized values) and arranges them into the
audit, detector-input and ground-truth tables, in the original row order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from attack_engine.attacks.executor import ExecutionSummary
from attack_engine.attacks.models import AttackResult
from attack_engine.dataset import temporal
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.exporting.contracts import AUDIT_EXTRA_COLUMNS, GROUND_TRUTH_COLUMNS
from attack_engine.exporting.models import ExportError


@dataclass(frozen=True)
class AttackedTarget:
    """The final per-row target column and its row-level provenance labels."""

    final_target: pd.Series
    is_attacked: pd.Series
    attack_type: pd.Series
    attack_id: pd.Series


def build_attacked_target(
    frame: pd.DataFrame, schema: DatasetSchema, summary: ExecutionSummary
) -> AttackedTarget:
    """Overlay committed modifications onto a copy of the clean target column.

    Only ``applied`` results contribute, and only at positions where
    ``changed_mask`` is true, so a zero-effect attack (applied but with no
    position actually changed) can never produce a positive label, and a
    rejected event can never modify a row.
    """
    target = schema.target_column
    final_target = frame[target].astype("float64").copy()
    is_attacked = pd.Series(False, index=frame.index)
    attack_type = pd.Series(pd.NA, index=frame.index, dtype="object")
    attack_id = pd.Series(pd.NA, index=frame.index, dtype="object")

    written: set[int] = set()
    for result in summary.applied_results:
        _apply_result(result, final_target, is_attacked, attack_type, attack_id, written)
    return AttackedTarget(final_target, is_attacked, attack_type, attack_id)


def _apply_result(
    result: AttackResult,
    final_target: pd.Series,
    is_attacked: pd.Series,
    attack_type: pd.Series,
    attack_id: pd.Series,
    written: set[int],
) -> None:
    labels = [
        label
        for label, changed in zip(result.row_labels, result.changed_mask, strict=True)
        if changed
    ]
    if not labels:
        return
    overlap = written & set(labels)
    if overlap:
        raise ExportError(
            f"row(s) {sorted(overlap)} would be modified by more than one committed attack "
            f"(at {result.window_id}); this should already be prevented by the executor"
        )
    values = [
        value
        for value, changed in zip(result.candidate_values, result.changed_mask, strict=True)
        if changed
    ]
    final_target.loc[labels] = values
    is_attacked.loc[labels] = True
    attack_type.loc[labels] = result.family.value
    attack_id.loc[labels] = result.window_id
    written.update(labels)


def assign_row_partitions(
    frame: pd.DataFrame, schema: DatasetSchema, partitions: PartitionPlan
) -> pd.Series:
    """Exactly the Phase 1 temporal-partition logic, reused so labels never diverge."""
    times_utc = temporal.to_utc(frame[schema.timestamps.column])
    return temporal.assign_partitions(times_utc, partitions, schema.timestamps.interval)


def build_audit_dataset(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    partitions: PartitionPlan,
    attacked: AttackedTarget,
    partition_labels: pd.Series,
) -> pd.DataFrame:
    """``uscrn_attacked.parquet``: every original column untouched, plus labels."""
    audit = frame.copy(deep=True)
    audit[f"{schema.target_column}_attacked"] = attacked.final_target
    audit["is_attacked"] = attacked.is_attacked
    audit["attack_type"] = attacked.attack_type
    audit["attack_id"] = attacked.attack_id
    audit["partition"] = partition_labels
    expected = (*schema.column_names, *AUDIT_EXTRA_COLUMNS)
    if tuple(audit.columns) != expected:
        raise ExportError("audit dataset columns drifted from the documented contract")
    return audit.reset_index(drop=True)


def build_ground_truth(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    attacked: AttackedTarget,
    partition_labels: pd.Series,
) -> pd.DataFrame:
    """``ground_truth.parquet``: one row per source observation, labels from final values."""
    clean_target = frame[schema.target_column].astype("float64")
    clean_finite = clean_target.notna()
    # Derived independently from the final values (not from the mask), per the
    # Phase 5 contract: labels must reflect actual differences, not reservations.
    is_attacked = clean_finite & (attacked.final_target != clean_target)
    if (is_attacked & ~attacked.is_attacked).any() or (attacked.is_attacked & ~is_attacked).any():
        raise ExportError(
            "value-derived is_attacked disagrees with the executor's changed mask; "
            "refusing to export an inconsistent ground truth"
        )
    ground_truth = pd.DataFrame(
        {
            "station": frame[schema.station_column].to_numpy(),
            "time": frame[schema.timestamps.column].to_numpy(),
            "partition": partition_labels.to_numpy(),
            "clean_target": clean_target.to_numpy(),
            "attacked_target": attacked.final_target.to_numpy(),
            "is_attacked": is_attacked.to_numpy(),
            "attack_type": attacked.attack_type.to_numpy(),
            "attack_id": attacked.attack_id.to_numpy(),
            "is_evaluable": clean_finite.to_numpy(),
        }
    )
    if tuple(ground_truth.columns) != GROUND_TRUTH_COLUMNS:
        raise ExportError("ground truth columns drifted from the documented contract")
    return ground_truth


def build_detector_input(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    attacked: AttackedTarget,
    partition_labels: pd.Series,
    context_columns: tuple[str, ...],
) -> pd.DataFrame:
    """``detector_input.parquet``: observable columns only, attacked value as the target."""
    columns = {
        schema.station_column: frame[schema.station_column].to_numpy(),
        schema.timestamps.column: frame[schema.timestamps.column].to_numpy(),
        "partition": partition_labels.to_numpy(),
        schema.target_column: attacked.final_target.to_numpy(),
    }
    for name in context_columns:
        columns[name] = frame[name].to_numpy()
    detector_input = pd.DataFrame(columns)
    forbidden = set(detector_input.columns) & {"is_attacked", "attack_type", "attack_id"}
    if forbidden:
        raise ExportError(f"detector input would leak privileged columns: {sorted(forbidden)}")
    return detector_input


def ground_truth_label_mask(
    ground_truth: pd.DataFrame, label_partitions: tuple[str, ...]
) -> np.ndarray:
    """Rows whose ground-truth labels the AI team may access (see HANDOVER.md)."""
    return ground_truth["partition"].isin(label_partitions).to_numpy()
