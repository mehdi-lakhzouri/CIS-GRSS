from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from attack_engine.config import ConfigError, load_config
from attack_engine.dataset.schema import ColumnKind, ColumnSpec, uscrn_hourly_schema

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dataset.yaml"


def test_project_config_loads() -> None:
    config = load_config(PROJECT_CONFIG)

    assert config.validation_profile.expected_column_count == 23
    assert config.expected_sha256 is not None
    assert config.partitions is not None
    assert [partition.name for partition in config.partitions.partitions] == [
        "train",
        "validation",
        "test",
    ]


def test_overlapping_partitions_in_config_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "validation_profile: {name: p}\n"
        "partitions:\n"
        "  - {name: a, start: '2022-01-01T00:00:00Z', end: '2024-06-01T00:00:00Z'}\n"
        "  - {name: b, start: '2024-01-01T00:00:00Z', end: '2025-01-01T00:00:00Z'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="overlap"):
        load_config(path)


@pytest.mark.parametrize(
    "content",
    [
        "validation_profile: [unclosed",
        "- just a list",
        "validation_profile: {name: p}\nexpected_sha256: nothex",
        "validation_profile: {name: p}\nunknown_key: 1",
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_uscrn_schema_roles() -> None:
    schema = uscrn_hourly_schema()

    assert schema.key_columns == ("station", "time")
    target = schema.column(schema.target_column)
    assert (target.min_value, target.max_value, target.unit) == (0.0, 0.70, "m3/m3")
    assert not schema.column("time").nullable
    assert schema.timestamps.position == "interval_end"
    assert len(schema.columns) == 23


def test_column_spec_rejects_inverted_range() -> None:
    with pytest.raises(ValidationError):
        ColumnSpec(name="x", kind=ColumnKind.FLOAT, min_value=1.0, max_value=0.0)


def test_column_spec_rejects_range_on_strings() -> None:
    with pytest.raises(ValidationError):
        ColumnSpec(name="x", kind=ColumnKind.STRING, min_value=0.0)
