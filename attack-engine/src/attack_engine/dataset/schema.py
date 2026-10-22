"""Declarative schema of the hourly USCRN observation table.

The schema states what a valid table looks like: columns, types, units, ranges,
observation keys and timestamp semantics. It contains no loading or validation
logic. Properties of one particular delivery (row count, station inventory) are
not part of the schema; they live in a ``ValidationProfile``.
"""

from __future__ import annotations

import re
from datetime import timedelta
from enum import StrEnum
from itertools import pairwise
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class ColumnKind(StrEnum):
    DATETIME = "datetime"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


NUMERIC_KINDS = frozenset({ColumnKind.INTEGER, ColumnKind.FLOAT})


class ColumnSpec(BaseModel):
    """Expected properties of one column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    kind: ColumnKind
    required: bool = True
    nullable: bool = True
    unit: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check_range(self) -> ColumnSpec:
        has_range = self.min_value is not None or self.max_value is not None
        if has_range and self.kind not in NUMERIC_KINDS:
            raise ValueError(f"column {self.name!r}: ranges only apply to numeric columns")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError(f"column {self.name!r}: min_value exceeds max_value")
        return self


class TimestampSemantics(BaseModel):
    """How observation timestamps are interpreted.

    USCRN hourly records are stamped at the END of the measured interval, in UTC:
    a record stamped ``t`` covers ``[t - interval, t)``. Timezone-naive source
    values are declared to be UTC.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str = "time"
    position: Literal["interval_end"] = "interval_end"
    naive_timezone: Literal["UTC"] = "UTC"
    interval: timedelta = timedelta(hours=1)
    # Source column expected to equal year(time - interval), if the table carries one.
    year_column: str | None = None

    @model_validator(mode="after")
    def _check_interval(self) -> TimestampSemantics:
        if self.interval <= timedelta(0):
            raise ValueError("interval must be positive")
        return self


class DatasetSchema(BaseModel):
    """Column definitions plus the roles (key, target, station, coordinates) they play."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    columns: tuple[ColumnSpec, ...] = Field(min_length=1)
    station_column: str
    target_column: str
    latitude_column: str
    longitude_column: str
    key_columns: tuple[str, ...] = Field(min_length=1)
    station_id_pattern: str
    timestamps: TimestampSemantics

    @model_validator(mode="after")
    def _check_references(self) -> DatasetSchema:
        names = [spec.name for spec in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("column names must be unique")
        referenced = {
            self.station_column,
            self.target_column,
            self.latitude_column,
            self.longitude_column,
            self.timestamps.column,
            *self.key_columns,
        }
        if self.timestamps.year_column is not None:
            referenced.add(self.timestamps.year_column)
        undefined = sorted(referenced - set(names))
        if undefined:
            raise ValueError(f"schema references undefined columns: {undefined}")
        for name in self.key_columns:
            spec = self.column(name)
            if not spec.required or spec.nullable:
                raise ValueError(f"key column {name!r} must be required and non-nullable")
        if not self.column(self.target_column).required:
            raise ValueError("the target column must be required")
        try:
            re.compile(self.station_id_pattern)
        except re.error as exc:
            raise ValueError(f"invalid station_id_pattern: {exc}") from exc
        return self

    def column(self, name: str) -> ColumnSpec:
        for spec in self.columns:
            if spec.name == name:
                return spec
        raise KeyError(name)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.columns)


class ValidationProfile(BaseModel):
    """Delivery-specific expectations layered on top of the generic schema.

    Every expectation is optional, so the same validator serves both the real
    reference delivery and small synthetic test fixtures.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    expected_row_count: int | None = Field(default=None, ge=0)
    expected_column_count: int | None = Field(default=None, ge=1)
    expected_stations: tuple[str, ...] | None = None
    # Every (station, observation year) must hold exactly one record per interval.
    require_complete_years: bool = False
    # Downgrade gaps in hourly continuity from errors to warnings.
    allow_temporal_gaps: bool = False


FIXTURE_PROFILE = ValidationProfile(name="test-fixture")


class PartitionSpec(BaseModel):
    """A chronological partition of measured intervals.

    A record belongs to the partition when ``start <= time - interval < end``,
    which for interval-end stamps is equivalent to ``start < time <= end``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _check_bounds(self) -> PartitionSpec:
        if self.start >= self.end:
            raise ValueError(f"partition {self.name!r}: start must precede end")
        return self


class PartitionPlan(BaseModel):
    """An ordered set of mutually exclusive chronological partitions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partitions: tuple[PartitionSpec, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _accept_sequence(cls, data: Any) -> Any:
        if isinstance(data, list | tuple):
            return {"partitions": data}
        return data

    @model_validator(mode="after")
    def _check_disjoint(self) -> PartitionPlan:
        names = [partition.name for partition in self.partitions]
        if len(names) != len(set(names)):
            raise ValueError("partition names must be unique")
        ordered = sorted(self.partitions, key=lambda partition: partition.start)
        for earlier, later in pairwise(ordered):
            if later.start < earlier.end:
                raise ValueError(f"partitions {earlier.name!r} and {later.name!r} overlap")
        return self


_SOIL_MOISTURE_UNIT = "m3/m3"
# Bounds used by the reference delivery's cleaning step; a project contract, not a
# universal property of all soils.
_SOIL_MOISTURE_BOUNDS = {"min_value": 0.0, "max_value": 0.70}
_TEMPERATURE_BOUNDS = {"min_value": -60.0, "max_value": 70.0}


def _soil_moisture(depth_cm: int, *, required: bool) -> ColumnSpec:
    return ColumnSpec(
        name=f"SOIL_MOISTURE_{depth_cm}",
        kind=ColumnKind.FLOAT,
        required=required,
        unit=_SOIL_MOISTURE_UNIT,
        description=f"Volumetric soil water content at {depth_cm} cm, hourly average",
        **_SOIL_MOISTURE_BOUNDS,
    )


def _temperature(name: str, description: str, *, required: bool) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        kind=ColumnKind.FLOAT,
        required=required,
        unit="degC",
        description=description,
        **_TEMPERATURE_BOUNDS,
    )


def uscrn_hourly_schema() -> DatasetSchema:
    """Schema of the cleaned USCRN hourly table (NOAA/NCEI Hourly02 derived)."""
    columns = (
        ColumnSpec(
            name="time",
            kind=ColumnKind.DATETIME,
            nullable=False,
            unit="UTC",
            description="End of the measured hour",
        ),
        ColumnSpec(
            name="station",
            kind=ColumnKind.STRING,
            nullable=False,
            description="USCRN station identifier (STATE_Location_Distance_Direction)",
        ),
        ColumnSpec(
            name="annee",
            kind=ColumnKind.INTEGER,
            nullable=False,
            description="Year of the measured interval start, year(time - 1 h)",
        ),
        ColumnSpec(
            name="LATITUDE",
            kind=ColumnKind.FLOAT,
            nullable=False,
            unit="degrees_north",
            min_value=-90.0,
            max_value=90.0,
        ),
        ColumnSpec(
            name="LONGITUDE",
            kind=ColumnKind.FLOAT,
            nullable=False,
            unit="degrees_east",
            min_value=-180.0,
            max_value=180.0,
        ),
        _temperature("T_HR_AVG", "Average air temperature over the hour", required=True),
        _temperature("T_MAX", "Maximum air temperature over the hour", required=False),
        _temperature("T_MIN", "Minimum air temperature over the hour", required=False),
        ColumnSpec(
            name="P_CALC",
            kind=ColumnKind.FLOAT,
            unit="mm",
            min_value=0.0,
            max_value=200.0,
            description="Precipitation total over the hour (an accumulation, not a rate)",
        ),
        ColumnSpec(
            name="SOLARAD",
            kind=ColumnKind.FLOAT,
            required=False,
            unit="W/m2",
            min_value=0.0,
            max_value=1600.0,
            description="Average global solar radiation over the hour",
        ),
        _temperature("SUR_TEMP", "Average infrared surface temperature", required=False),
        ColumnSpec(
            name="RH_HR_AVG",
            kind=ColumnKind.FLOAT,
            unit="%",
            min_value=0.0,
            max_value=100.0,
            description="Average relative humidity over the hour",
        ),
        _soil_moisture(5, required=True),
        _soil_moisture(10, required=True),
        _soil_moisture(20, required=True),
        _soil_moisture(50, required=False),
        _soil_moisture(100, required=False),
        _temperature("SOIL_TEMP_5", "Soil temperature at 5 cm", required=True),
        _temperature("SOIL_TEMP_10", "Soil temperature at 10 cm", required=True),
        _temperature("SOIL_TEMP_20", "Soil temperature at 20 cm", required=False),
        _temperature("SOIL_TEMP_50", "Soil temperature at 50 cm", required=False),
        _temperature("SOIL_TEMP_100", "Soil temperature at 100 cm", required=False),
        ColumnSpec(
            name="flag_incoherence_verticale",
            kind=ColumnKind.BOOLEAN,
            required=False,
            nullable=False,
            description="Vertical-coherence flag derived by the delivery's cleaning step",
        ),
    )
    return DatasetSchema(
        name="uscrn-hourly",
        columns=columns,
        station_column="station",
        target_column="SOIL_MOISTURE_5",
        latitude_column="LATITUDE",
        longitude_column="LONGITUDE",
        key_columns=("station", "time"),
        station_id_pattern=r"[A-Z]{2}_[A-Za-z]+(?:_[A-Za-z0-9]+)*",
        timestamps=TimestampSemantics(column="time", year_column="annee"),
    )
