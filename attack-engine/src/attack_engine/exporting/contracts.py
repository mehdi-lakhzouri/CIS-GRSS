"""The explicit, reviewable detector-input allowlist and the dataset contract builder.

Detector-visible features are never inferred from "all available DataFrame
columns": they are this fixed allowlist, derived from the dataset schema but
excluding the target's clean value and every column that is itself derived from
the clean reference (and could therefore leak attack labels).
"""

from __future__ import annotations

from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.exporting.models import DatasetContract, ExportConfiguration

# Columns that must never reach the detector, with the reason they are privileged.
PRIVILEGED_COLUMNS: dict[str, str] = {
    "is_attacked": "ground-truth attack label",
    "attack_type": "ground-truth attack family label",
    "attack_id": "ground-truth event identifier",
    "flag_incoherence_verticale": "derived from the clean reference; could leak attack labels",
}

# Temporal/identifying metadata the detector is explicitly allowed to see.
APPROVED_TEMPORAL_COLUMNS: tuple[str, ...] = ("annee", "partition")

AUDIT_EXTRA_COLUMNS: tuple[str, ...] = (
    "SOIL_MOISTURE_5_attacked",
    "is_attacked",
    "attack_type",
    "attack_id",
    "partition",
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "attack_id",
    "family",
    "station",
    "partition",
    "window_start",
    "window_end_exclusive",
    "length",
    "status",
    "requested_capacity",
    "reserved_positions",
    "modified_count",
    "rejection_reason",
    "requested_magnitude",
    "sign",
    "mean_abs_delta",
    "max_abs_delta",
    "saturated_count",
    "replay_source_first_time",
    "replay_source_last_time",
    "replay_source_age_hours",
)

GROUND_TRUTH_COLUMNS: tuple[str, ...] = (
    "station",
    "time",
    "partition",
    "clean_target",
    "attacked_target",
    "is_attacked",
    "attack_type",
    "attack_id",
    "is_evaluable",
)


def default_detector_context_columns(schema: DatasetSchema) -> tuple[str, ...]:
    """The project default allowlist: every schema column except keys, the target
    itself, the year column (superseded by ``partition``) and privileged columns.
    """
    excluded = {
        schema.target_column,
        schema.station_column,
        schema.timestamps.column,
        schema.timestamps.year_column,
        *PRIVILEGED_COLUMNS,
    }
    return tuple(name for name in schema.column_names if name not in excluded)


def resolve_detector_context_columns(
    schema: DatasetSchema, config: ExportConfiguration
) -> tuple[str, ...]:
    """Resolve the configured or default allowlist, rejecting privileged columns."""
    if config.detector_context_columns is None:
        return default_detector_context_columns(schema)
    requested = config.detector_context_columns
    forbidden = (set(requested) & set(PRIVILEGED_COLUMNS)) | (
        {schema.target_column} & set(requested)
    )
    if forbidden:
        raise ValueError(
            f"detector_context_columns must not include privileged or target columns: "
            f"{sorted(forbidden)}"
        )
    unknown = sorted(set(requested) - set(schema.column_names))
    if unknown:
        raise ValueError(f"detector_context_columns references unknown schema columns: {unknown}")
    return requested


def build_dataset_contract(
    schema: DatasetSchema,
    partitions: PartitionPlan,
    config: ExportConfiguration,
    context_columns: tuple[str, ...],
) -> DatasetContract:
    detector_input_columns = (
        schema.station_column,
        schema.timestamps.column,
        "partition",
        schema.target_column,
        *context_columns,
    )
    return DatasetContract(
        key_columns=(schema.station_column, schema.timestamps.column),
        target_column=schema.target_column,
        attacked_target_column=f"{schema.target_column}_attacked",
        source_columns=schema.column_names,
        audit_extra_columns=AUDIT_EXTRA_COLUMNS,
        detector_input_columns=detector_input_columns,
        detector_context_columns=context_columns,
        ground_truth_columns=GROUND_TRUTH_COLUMNS,
        manifest_columns=MANIFEST_COLUMNS,
        excluded_from_detector=dict(PRIVILEGED_COLUMNS),
        partitions=tuple(partition.name for partition in partitions.partitions),
        detector_label_partitions=config.detector_label_partitions,
    )
