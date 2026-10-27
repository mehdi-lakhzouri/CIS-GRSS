"""Reading the detector release, deterministic writing, and hashing.

The only dataset this package ever opens is ``detector_input.parquet`` (or a file
with the same observation columns). Label and evaluator files are refused by name,
and only the columns listed here are read, so a label column can never leak in.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEY_COLUMNS = ("station", "time")
SIGNAL_COLUMNS = ("SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC")
OPTIONAL_COLUMNS = ("partition",)

# Evaluator/label artifacts of the attack engine. Opening any of them is refused.
FORBIDDEN_FILE_STEMS = frozenset(
    {
        "row_labels",
        "cell_labels",
        "ground_truth",
        "ground_truth_cells",
        "uscrn_attacked",
        "attack_manifest",
    }
)

_US_PER_HOUR = 3_600_000_000


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_detector_input(path: Path) -> pd.DataFrame:
    """Read keys, the three observed signals and (if present) ``partition``.

    Row order is preserved. Raises ``ValueError`` on a forbidden file, missing
    columns, duplicate ``(station, time)`` keys or timestamps off the hour.
    """
    if path.stem in FORBIDDEN_FILE_STEMS:
        raise ValueError(f"refusing to read label/evaluator file: {path.name}")
    available = set(pq.read_schema(path).names)
    missing = [c for c in (*KEY_COLUMNS, *SIGNAL_COLUMNS) if c not in available]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    columns = [*KEY_COLUMNS, *SIGNAL_COLUMNS, *(c for c in OPTIONAL_COLUMNS if c in available)]
    frame = pq.read_table(path, columns=columns).to_pandas()
    validate_frame(frame)
    return frame


def validate_frame(frame: pd.DataFrame) -> None:
    """Check key uniqueness and hourly timestamps."""
    if not pd.api.types.is_datetime64_any_dtype(frame["time"]):
        raise ValueError("column 'time' must be a datetime column")
    if frame["time"].isna().any() or frame["station"].isna().any():
        raise ValueError("keys (station, time) must not be missing")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("duplicate (station, time) keys in input")
    epoch_hours(frame["time"])


def epoch_hours(time: pd.Series) -> np.ndarray:
    """Whole hours since the Unix epoch (UTC). Naive timestamps are taken as UTC."""
    values = time
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        values = values.dt.tz_convert("UTC").dt.tz_localize(None)
    micros = values.to_numpy(dtype="datetime64[us]").astype(np.int64)
    if np.any(micros % _US_PER_HOUR):
        raise ValueError("timestamps must fall exactly on the hour")
    return micros // _US_PER_HOUR


def write_parquet_deterministic(frame: pd.DataFrame, path: Path) -> None:
    """Write Parquet with fixed settings and no pandas metadata, atomically."""
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=3, row_group_size=65_536)
    tmp.replace(path)


def dumps_json(payload: Any) -> bytes:
    """Stable JSON bytes: 2-space indent, LF line endings, trailing newline."""
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_bytes_atomic(data: bytes, path: Path) -> None:
    """Write bytes via a temporary sibling and rename (no partial files)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
