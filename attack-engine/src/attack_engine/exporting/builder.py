"""Build export and audience-specific tables from an already-committed execution.

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
from attack_engine.attacks.models import AttackResult, ColumnModification
from attack_engine.dataset import temporal
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.exporting.contracts import (
    AUDIT_EXTRA_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    VARIABLE_GROUND_TRUTH_COLUMNS,
)
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


def build_cell_ground_truth(ground_truth: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Variable-level ground truth: one row per attacked ``(station, time, variable)`` cell.

    Legacy foundation projection. It projects row-level ground truth onto the per-variable
    (cell) schema in :data:`VARIABLE_GROUND_TRUTH_COLUMNS`. The current single-target run
    attacks only ``target_column``, so each attacked row yields exactly one cell; a future
    multi-variable run would contribute one row per attacked variable of a row. This table is
    retained for compatibility; the pipeline writes full truth via build_variable_ground_truth.
    """
    attacked = ground_truth[ground_truth["is_attacked"].to_numpy()]
    cells = pd.DataFrame(
        {
            "station": attacked["station"].to_numpy(),
            "time": attacked["time"].to_numpy(),
            "partition": attacked["partition"].to_numpy(),
            "column": target_column,
            "clean_value": attacked["clean_target"].to_numpy(),
            "attacked_value": attacked["attacked_target"].to_numpy(),
            "is_attacked": attacked["is_attacked"].to_numpy(),
            "attack_type": attacked["attack_type"].to_numpy(),
            "attack_id": attacked["attack_id"].to_numpy(),
            "is_evaluable": attacked["is_evaluable"].to_numpy(),
        }
    )
    if tuple(cells.columns) != VARIABLE_GROUND_TRUTH_COLUMNS:
        raise ExportError("cell ground truth columns drifted from the documented contract")
    return cells.reset_index(drop=True)


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


def build_attacked_columns(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    summary: ExecutionSummary,
    target_columns: tuple[str, ...],
) -> dict[str, AttackedTarget]:
    """Overlay every selected channel, independently checking exact masks and cell conflicts."""
    attacked = {
        name: AttackedTarget(
            frame[name].astype(float).copy(),
            pd.Series(False, index=frame.index),
            pd.Series(pd.NA, index=frame.index, dtype=object),
            pd.Series(pd.NA, index=frame.index, dtype=object),
        )
        for name in target_columns
    }
    written: set[tuple[int, str]] = set()
    for result in summary.results:
        if not result.applied:
            if result.modified_cells:
                raise ExportError(f"rejected event {result.window_id} declares modified cells")
            continue
        for column in result.columns:
            if column.column not in attacked:
                raise ExportError(
                    f"event {result.window_id} writes unconfigured target {column.column}"
                )
            _validate_column(frame, schema, result, column)
            keys = column.cell_keys()
            if written.intersection(keys) or len(keys) != len(set(keys)):
                raise ExportError(
                    f"cell would be modified by more than one event: {result.window_id}"
                )
            written.update(keys)
            labels = list(column.modified_labels)
            values = np.asarray(column.candidate_values)[np.asarray(column.changed_mask, bool)]
            overlay = attacked[column.column]
            overlay.final_target.loc[labels] = values
            overlay.is_attacked.loc[labels] = True
            overlay.attack_type.loc[labels] = result.family.value
            overlay.attack_id.loc[labels] = result.window_id
    return attacked


def _validate_column(frame, schema, result: AttackResult, column: ColumnModification) -> None:
    labels = list(column.row_labels)
    if (
        len(set(labels)) != len(labels)
        or not frame.index.is_unique
        or np.any(frame.index.get_indexer(labels) < 0)
    ):
        raise ExportError(f"invalid row labels in {result.window_id}")
    clean = np.asarray(column.clean_values, dtype=float)
    final = np.asarray(column.candidate_values, dtype=float)
    mask = np.asarray(column.changed_mask, bool)
    if len(clean) != len(labels) or final.shape != clean.shape or mask.shape != clean.shape:
        raise ExportError(f"invalid column array shapes in {result.window_id}")
    if not np.array_equal(
        clean, frame.loc[labels, column.column].to_numpy(dtype=float), equal_nan=True
    ):
        raise ExportError(
            f"clean values disagree with source in {result.window_id}/{column.column}"
        )
    actual = np.isfinite(clean) & np.isfinite(final) & (clean != final)
    if (
        not np.array_equal(mask, actual)
        or not np.array_equal(np.isnan(clean), np.isnan(final))
        or np.any(np.isfinite(clean) & ~np.isfinite(final))
    ):
        raise ExportError(f"exact changed mask or missing values disagree in {result.window_id}")
    if set(frame.loc[labels, schema.station_column]) != {result.station}:
        raise ExportError(f"event station disagrees with source in {result.window_id}")


def build_variable_ground_truth(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    attacked: dict[str, AttackedTarget],
    partition_labels: pd.Series,
) -> pd.DataFrame:
    """Full cell truth: one row per source observation and selected target, including negatives."""
    tables = []
    for name, overlay in attacked.items():
        table = build_ground_truth(
            frame,
            schema.model_copy(update={"target_column": name}),
            overlay,
            partition_labels,
        ).rename(columns={"clean_target": "clean_value", "attacked_target": "attacked_value"})
        table.insert(3, "column", name)
        table["is_evaluable"] = np.isfinite(table["clean_value"])
        tables.append(table.loc[:, VARIABLE_GROUND_TRUTH_COLUMNS])
    return pd.concat(tables, ignore_index=True)


def build_multivariable_ground_truth(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    attacked: dict[str, AttackedTarget],
    partition_labels: pd.Series,
) -> pd.DataFrame:
    """Row union, retaining primary compatibility fields and aligned values for every target."""
    names = tuple(attacked)
    primary = names[0]
    truth = build_ground_truth(
        frame,
        schema.model_copy(update={"target_column": primary}),
        attacked[primary],
        partition_labels,
    )
    clean = frame.loc[:, names].to_numpy(dtype=float)
    final = np.column_stack([overlay.final_target.to_numpy() for overlay in attacked.values()])
    changed = np.isfinite(clean) & np.isfinite(final) & (clean != final)
    finite = np.isfinite(clean)
    ids, families = [set() for _ in range(len(frame))], [set() for _ in range(len(frame))]
    for overlay in attacked.values():
        for pos in np.flatnonzero(overlay.is_attacked.to_numpy()):
            ids[pos].add(str(overlay.attack_id.iloc[pos]))
            families[pos].add(str(overlay.attack_type.iloc[pos]))
    truth["is_attacked"] = changed.any(axis=1)
    truth["is_evaluable"] = finite.any(axis=1)
    truth["attack_id"] = [next(iter(values)) if len(values) == 1 else None for values in ids]
    truth["attack_type"] = [
        next(iter(values)) if len(values) == 1 else ("multiple" if values else None)
        for values in families
    ]
    truth["clean_values"] = clean.tolist()
    truth["attacked_values"] = final.tolist()
    truth["changed_columns"] = [
        [name for name, flag in zip(names, row, strict=True) if flag] for row in changed
    ]
    truth["evaluable_columns"] = [
        [name for name, flag in zip(names, row, strict=True) if flag] for row in finite
    ]
    truth["modified_cells"] = changed.sum(axis=1)
    truth["attack_types"] = [sorted(values) for values in families]
    truth["attack_ids"] = [sorted(values) for values in ids]
    return truth


def build_variable_audit_dataset(frame, attacked, ground_truth, partition_labels) -> pd.DataFrame:
    audit = frame.copy(deep=True)
    for name, overlay in attacked.items():
        audit[f"{name}_attacked"] = overlay.final_target
    for name in ("is_attacked", "attack_type", "attack_id"):
        audit[name] = ground_truth[name].to_numpy()
    audit["partition"] = partition_labels
    return audit.reset_index(drop=True)


def build_variable_detector_input(frame, schema, attacked, partition_labels, context_columns):
    columns = {
        schema.station_column: frame[schema.station_column].to_numpy(),
        schema.timestamps.column: frame[schema.timestamps.column].to_numpy(),
        "partition": partition_labels.to_numpy(),
        **{name: overlay.final_target.to_numpy() for name, overlay in attacked.items()},
    }
    for name in context_columns:
        columns[name] = (
            attacked[name].final_target if name in attacked else frame[name]
        ).to_numpy()
    return pd.DataFrame(columns)


def build_detector_labels(truth: pd.DataFrame, allowed: tuple[str, ...], *, cells=False):
    keys = ["station", "time", "partition", *(["column"] if cells else [])]
    columns = [*keys, "is_attacked", "attack_type", "attack_id", "is_evaluable"]
    if not cells and "changed_columns" in truth:
        columns += ["changed_columns", "modified_cells", "attack_types", "attack_ids"]
    return truth.loc[ground_truth_label_mask(truth, allowed), columns].reset_index(drop=True)
