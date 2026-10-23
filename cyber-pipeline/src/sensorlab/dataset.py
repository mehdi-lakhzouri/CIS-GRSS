"""Read exactly two public release files; never inspect adjacent label/audit files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict, model_validator

from sensorlab.registry import MEASUREMENT_COLUMNS, RELEASE_COLUMNS, STATION_SENSORS
from sensorlab.telemetry import Measurements


class DatasetError(ValueError):
    """Public release or selection violates the supported contract."""


class ReleaseContract(BaseModel):
    # Ignore documentation about private artifacts; never open those artifacts.
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["1.0", "2.0"]
    key_columns: list[str]
    detector_input_columns: list[str]

    @model_validator(mode="after")
    def safe_columns(self) -> ReleaseContract:
        columns = self.detector_input_columns
        if len(columns) != len(set(columns)) or set(columns) != RELEASE_COLUMNS:
            raise ValueError("detector columns must match the supported observable allowlist")
        if set(self.key_columns) != {"station", "time"} or len(self.key_columns) != 2:
            raise ValueError("keys must be station and time")
        return self


@dataclass(frozen=True)
class Observation:
    station: str
    time: datetime
    measurements: Measurements


def parse_utc(value: str) -> datetime:
    """Require an explicit timezone on user date filters, then normalize to UTC."""
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetError(f"invalid ISO-8601 timestamp: {value}") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise DatasetError("date filters require a timezone, for example 2025-01-01T01:00:00Z")
    return result.astimezone(UTC)


class DetectorDataset:
    def __init__(self, frame: pd.DataFrame, contract: ReleaseContract):
        self._frame = frame
        self.contract = contract

    @classmethod
    def load(cls, release: Path) -> DetectorDataset:
        contract_path = release / "data_contract.json"
        input_path = release / "detector_input.parquet"
        contract = ReleaseContract.model_validate_json(contract_path.read_text(encoding="utf-8"))
        columns = pq.read_schema(input_path).names
        if len(columns) != len(set(columns)) or set(columns) != RELEASE_COLUMNS:
            raise DatasetError("Parquet columns differ from the observable contract")
        frame = pd.read_parquet(input_path, columns=contract.detector_input_columns)
        if frame.empty:
            raise DatasetError("release is empty")
        if frame["station"].isna().any() or set(frame["station"]) != set(STATION_SENSORS):
            raise DatasetError("release must contain exactly the six registered stations")
        if not pd.api.types.is_datetime64_any_dtype(frame["time"]):
            raise DatasetError("time must be a Parquet timestamp")
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        if frame["time"].isna().any() or not (frame["time"] == frame["time"].dt.floor("h")).all():
            raise DatasetError("timestamps must be nonmissing hourly interval ends")
        if frame.duplicated(["station", "time"]).any():
            raise DatasetError("duplicate station/time observation")
        if frame["partition"].isna().any() or not set(frame["partition"]) <= {
            "train",
            "validation",
            "test",
        }:
            raise DatasetError("invalid public partition metadata")
        for column in MEASUREMENT_COLUMNS:
            if not is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
                raise DatasetError(f"measurement {column} must be numeric")
            nonmissing = frame[column].dropna()
            if nonmissing.isin([float("inf"), float("-inf")]).any():
                raise DatasetError(f"measurement {column} contains infinity")
        return cls(frame.sort_values(["time", "station"], kind="stable"), contract)

    def select(
        self,
        stations: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> DetectorDataset:
        if stations is not None and (
            not stations
            or len(stations) != len(set(stations))
            or not set(stations) <= set(STATION_SENSORS)
        ):
            raise DatasetError("select unique registered station names")
        for bound in (start, end):
            if bound is not None and (bound.tzinfo is None or bound.utcoffset() is None):
                raise DatasetError("date filters must be timezone-aware")
        if start is not None and end is not None and start >= end:
            raise DatasetError("start must precede end (exclusive)")
        mask = pd.Series(True, index=self._frame.index)
        if stations is not None:
            mask &= self._frame["station"].isin(stations)
        if start is not None:
            mask &= self._frame["time"] >= start
        if end is not None:
            mask &= self._frame["time"] < end
        selected = self._frame.loc[mask].copy()
        if selected.empty:
            raise DatasetError("selection contains no observations")
        return DetectorDataset(selected, self.contract)

    def observations(self):
        columns = ["station", "time", *MEASUREMENT_COLUMNS]
        for row in self._frame[columns].itertuples(index=False, name=None):
            values = {
                key: None if pd.isna(value) else float(value)
                for key, value in zip(MEASUREMENT_COLUMNS, row[2:], strict=True)
            }
            yield Observation(row[0], row[1].to_pydatetime(), Measurements(**values))

    @property
    def stations(self) -> frozenset[str]:
        return frozenset(self._frame["station"].unique())

    def summary(self) -> dict:
        return {
            "rows": len(self._frame),
            "stations": self._frame.groupby("station").size().to_dict(),
            "first_observation": self._frame["time"].min().isoformat(),
            "last_observation": self._frame["time"].max().isoformat(),
            "measurement_columns": list(MEASUREMENT_COLUMNS),
            "missing_measurements": {
                name: int(self._frame[name].isna().sum()) for name in MEASUREMENT_COLUMNS
            },
        }

    def summary_json(self) -> str:
        return json.dumps(self.summary(), indent=2)
