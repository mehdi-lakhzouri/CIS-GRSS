from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attack_engine.dataset.loader import (
    DatasetLoader,
    DatasetNotFoundError,
    DatasetReadError,
    UnsupportedFormatError,
)
from attack_engine.dataset.schema import DatasetSchema


@pytest.fixture
def loader(schema: DatasetSchema) -> DatasetLoader:
    return DatasetLoader(schema=schema)


def test_loads_valid_parquet_with_schema_metadata_and_hash(
    loader: DatasetLoader, parquet_file: Path, observations: pd.DataFrame
) -> None:
    dataset = loader.load(parquet_file)

    pd.testing.assert_frame_equal(dataset.frame, observations, check_exact=True)
    assert dataset.schema is loader.schema
    assert dataset.source_sha256 == hashlib.sha256(parquet_file.read_bytes()).hexdigest()
    assert dataset.metadata.row_count == len(observations)
    assert dataset.metadata.column_count == len(observations.columns)
    assert dataset.metadata.size_bytes == parquet_file.stat().st_size
    fields = {field.name: field for field in dataset.metadata.arrow_fields}
    assert fields["time"].type == "timestamp[us]"


def test_preserves_missing_values_and_full_precision(
    loader: DatasetLoader, tmp_path: Path, observations: pd.DataFrame
) -> None:
    frame = observations.copy()
    frame.loc[3, "SOIL_MOISTURE_5"] = np.nan
    frame.loc[4, "SOIL_MOISTURE_5"] = 0.12345678901234567
    frame.loc[5, "P_CALC"] = np.nan
    path = tmp_path / "precision.parquet"
    frame.to_parquet(path, index=False)

    loaded = loader.load(path).frame

    assert np.isnan(loaded.loc[3, "SOIL_MOISTURE_5"])
    assert loaded.loc[4, "SOIL_MOISTURE_5"] == 0.12345678901234567
    assert np.isnan(loaded.loc[5, "P_CALC"]), "missing context must not become zero"
    pd.testing.assert_frame_equal(loaded, frame, check_exact=True)


def test_missing_file_raises(loader: DatasetLoader, tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match="not found"):
        loader.load(tmp_path / "absent.parquet")


def test_directory_is_not_a_dataset(loader: DatasetLoader, tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError):
        loader.load(tmp_path)


def test_unsupported_extension_raises(
    loader: DatasetLoader, tmp_path: Path, observations: pd.DataFrame
) -> None:
    path = tmp_path / "observations.csv"
    observations.to_csv(path, index=False)
    with pytest.raises(UnsupportedFormatError, match=r"\.csv"):
        loader.load(path)


def test_corrupted_parquet_raises(loader: DatasetLoader, tmp_path: Path) -> None:
    path = tmp_path / "corrupted.parquet"
    path.write_bytes(b"PAR1 this is not a parquet payload")
    with pytest.raises(DatasetReadError, match="cannot read"):
        loader.load(path)


def test_truncated_parquet_raises(
    loader: DatasetLoader, tmp_path: Path, parquet_file: Path
) -> None:
    truncated = tmp_path / "truncated.parquet"
    truncated.write_bytes(parquet_file.read_bytes()[:-64])
    with pytest.raises(DatasetReadError):
        loader.load(truncated)


def test_source_file_is_left_unchanged(loader: DatasetLoader, parquet_file: Path) -> None:
    content_before = parquet_file.read_bytes()
    stat_before = parquet_file.stat()

    loader.load(parquet_file)

    stat_after = parquet_file.stat()
    assert parquet_file.read_bytes() == content_before
    assert (stat_after.st_size, stat_after.st_mtime_ns) == (
        stat_before.st_size,
        stat_before.st_mtime_ns,
    )
    assert sorted(path.name for path in parquet_file.parent.iterdir()) == [parquet_file.name]
