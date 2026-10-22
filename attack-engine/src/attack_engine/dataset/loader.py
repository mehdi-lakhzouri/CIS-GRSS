"""Read-only loading of an observation table with provenance metadata.

The loader never cleans, imputes, sorts or converts observations: the returned
DataFrame holds exactly the columns, values, missing values and precision stored
in the source file, which is only ever opened for reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from attack_engine.dataset.schema import DatasetSchema
from attack_engine.utils.hashing import DEFAULT_CHUNK_SIZE, sha256_file


class DatasetError(Exception):
    """Base class for dataset input errors."""


class DatasetNotFoundError(DatasetError):
    """The dataset path does not point to an existing file."""


class UnsupportedFormatError(DatasetError):
    """The dataset file extension is not supported."""


class DatasetReadError(DatasetError):
    """The dataset file exists but cannot be read as a table."""


class ArrowField(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    nullable: bool


class SourceMetadata(BaseModel):
    """Provenance of a loaded file, captured before any processing."""

    model_config = ConfigDict(frozen=True)

    path: str
    file_name: str
    format: str
    size_bytes: int
    modified_at: datetime
    sha256: str
    row_count: int
    column_count: int
    row_group_count: int
    created_by: str | None
    arrow_fields: tuple[ArrowField, ...]


@dataclass(frozen=True)
class LoadedDataset:
    frame: pd.DataFrame
    schema: DatasetSchema
    metadata: SourceMetadata

    @property
    def source_sha256(self) -> str:
        return self.metadata.sha256


@dataclass(frozen=True)
class DatasetLoader:
    schema: DatasetSchema
    hash_chunk_size: int = DEFAULT_CHUNK_SIZE

    SUPPORTED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".parquet"})

    def load(self, path: Path | str) -> LoadedDataset:
        source = Path(path)
        if not source.is_file():
            raise DatasetNotFoundError(f"dataset file not found: {source}")
        extension = source.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise UnsupportedFormatError(
                f"unsupported dataset format {extension or '(none)'!r} for {source}; "
                f"supported: {supported}"
            )

        before = source.stat()
        digest = sha256_file(source, self.hash_chunk_size)
        try:
            with pq.ParquetFile(source) as parquet:
                table = parquet.read()
                file_metadata = parquet.metadata
        except (pa.ArrowException, OSError) as exc:
            raise DatasetReadError(f"cannot read Parquet file {source}: {exc}") from exc
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise DatasetReadError(f"dataset file changed while being loaded: {source}")

        metadata = SourceMetadata(
            path=str(source.resolve()),
            file_name=source.name,
            format="parquet",
            size_bytes=before.st_size,
            modified_at=datetime.fromtimestamp(before.st_mtime, tz=UTC),
            sha256=digest,
            row_count=table.num_rows,
            column_count=table.num_columns,
            row_group_count=file_metadata.num_row_groups,
            created_by=file_metadata.created_by,
            arrow_fields=tuple(
                ArrowField(name=field.name, type=str(field.type), nullable=field.nullable)
                for field in table.schema
            ),
        )
        return LoadedDataset(frame=table.to_pandas(), schema=self.schema, metadata=metadata)
