from __future__ import annotations

import pytest

from attack_engine.exporting.contracts import (
    PRIVILEGED_COLUMNS,
    build_dataset_contract,
    default_detector_context_columns,
    resolve_detector_context_columns,
)
from attack_engine.exporting.models import ExportConfiguration


def test_default_allowlist_excludes_target_keys_and_privileged_columns(schema) -> None:
    allowlist = default_detector_context_columns(schema)

    assert schema.target_column not in allowlist
    assert schema.station_column not in allowlist
    assert schema.timestamps.column not in allowlist
    assert schema.timestamps.year_column not in allowlist
    assert "flag_incoherence_verticale" not in allowlist
    assert "SOIL_MOISTURE_10" in allowlist
    assert "T_HR_AVG" in allowlist


def test_resolve_rejects_privileged_override(schema) -> None:
    config = ExportConfiguration(output_root="unused", detector_context_columns=("is_attacked",))
    with pytest.raises(ValueError, match="privileged"):
        resolve_detector_context_columns(schema, config)


def test_resolve_rejects_target_column_override(schema) -> None:
    config = ExportConfiguration(
        output_root="unused", detector_context_columns=(schema.target_column,)
    )
    with pytest.raises(ValueError, match="privileged"):
        resolve_detector_context_columns(schema, config)


def test_resolve_rejects_unknown_column(schema) -> None:
    config = ExportConfiguration(output_root="unused", detector_context_columns=("NOT_A_COLUMN",))
    with pytest.raises(ValueError, match="unknown"):
        resolve_detector_context_columns(schema, config)


def test_resolve_accepts_custom_allowlist(schema) -> None:
    config = ExportConfiguration(output_root="unused", detector_context_columns=("T_HR_AVG",))
    assert resolve_detector_context_columns(schema, config) == ("T_HR_AVG",)


def test_build_dataset_contract_documents_the_allowlist(schema, partition_plan) -> None:
    config = ExportConfiguration(output_root="unused")
    columns = default_detector_context_columns(schema)

    contract = build_dataset_contract(schema, partition_plan, config, columns)

    assert contract.detector_context_columns == columns
    assert set(PRIVILEGED_COLUMNS) == set(contract.excluded_from_detector)
    assert contract.partitions == ("train", "validation", "test")
    assert contract.detector_label_partitions == ("train", "validation")
    assert contract.detector_input_columns.count(contract.target_column) == 1
    assert "is_attacked" not in contract.detector_input_columns
