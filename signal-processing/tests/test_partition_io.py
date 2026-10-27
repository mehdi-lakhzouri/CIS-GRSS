"""Partition rule, input validation and label-file refusal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_processing.io import epoch_hours, read_detector_input
from signal_processing.partition import DEFAULT_RULES, assign_partitions, load_partition_rules


def test_partition_boundaries_follow_interval_end_rule() -> None:
    times = pd.Series(
        pd.to_datetime(
            ["2022-01-01 01:00", "2024-01-01 00:00", "2024-01-01 01:00", "2026-01-01 00:00"]
        )
    )
    parts = assign_partitions(epoch_hours(times), DEFAULT_RULES)
    assert parts.tolist() == ["train", "train", "validation", "test"]


def test_engine_config_matches_defaults() -> None:
    rules, source = load_partition_rules()
    if source == "built-in defaults":
        pytest.skip("engine configuration not present")
    assert rules == DEFAULT_RULES
    assert not Path(source).is_absolute()


def test_rejects_label_files_by_name(tmp_path: Path, frame: pd.DataFrame) -> None:
    path = tmp_path / "row_labels.parquet"
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="refusing"):
        read_detector_input(path)


def test_rejects_duplicate_keys_and_off_hour_times(tmp_path: Path, frame: pd.DataFrame) -> None:
    path = tmp_path / "detector_input.parquet"
    pd.concat([frame.head(3), frame.head(1)]).to_parquet(path)
    with pytest.raises(ValueError, match="duplicate"):
        read_detector_input(path)
    shifted = frame.head(3).assign(time=frame["time"].head(3) + pd.Timedelta(minutes=30))
    shifted.to_parquet(path)
    with pytest.raises(ValueError, match="on the hour"):
        read_detector_input(path)


def test_reads_only_allowed_columns(tmp_path: Path, frame: pd.DataFrame) -> None:
    path = tmp_path / "detector_input.parquet"
    frame.assign(is_attacked=False, T_HR_AVG=np.nan).to_parquet(path)
    loaded = read_detector_input(path)
    assert "is_attacked" not in loaded.columns
    assert list(loaded.columns) == [
        "station",
        "time",
        "SOIL_MOISTURE_5",
        "SOIL_MOISTURE_10",
        "P_CALC",
    ]
