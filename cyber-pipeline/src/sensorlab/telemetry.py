"""Versioned, strict public telemetry contract. All measurement fields are required."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from sensorlab.registry import SENSOR_STATIONS


class Measurements(BaseModel):
    """Original numeric observations; absent readings serialize as null, never NaN."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    SOIL_MOISTURE_5: float | None
    SOIL_MOISTURE_10: float | None
    P_CALC: float | None
    LATITUDE: float | None
    LONGITUDE: float | None
    T_HR_AVG: float | None
    T_MAX: float | None
    T_MIN: float | None
    SOLARAD: float | None
    SUR_TEMP: float | None
    RH_HR_AVG: float | None
    SOIL_MOISTURE_20: float | None
    SOIL_MOISTURE_50: float | None
    SOIL_MOISTURE_100: float | None
    SOIL_TEMP_5: float | None
    SOIL_TEMP_10: float | None
    SOIL_TEMP_20: float | None
    SOIL_TEMP_50: float | None
    SOIL_TEMP_100: float | None


class TelemetryFields(BaseModel):
    """Structural wire fields; identity binding is checked separately by the Crypto Gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    sensor_id: str = Field(strict=True)
    station: str = Field(strict=True)
    session_id: UUID
    sequence: int = Field(ge=1, strict=True)
    observation_time: AwareDatetime = Field(
        json_schema_extra={"pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:00:00(?:\.0+)?(?:Z|\+00:00)$"}
    )
    sent_at: AwareDatetime = Field(
        json_schema_extra={"pattern": r"^\d{4}-\d{2}-\d{2}T.*(?:Z|\+00:00)$"}
    )
    measurements: Measurements

    @field_validator("observation_time", "sent_at", mode="before")
    @classmethod
    def timestamp_input(cls, value: object) -> datetime:
        if isinstance(value, str) and "T" in value:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, datetime):
            return value
        raise ValueError("timestamps must be ISO-8601 date-time strings or aware datetimes")

    @field_validator("observation_time", "sent_at")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("timestamps must use UTC (Z or +00:00)")
        return value.astimezone(UTC)

    @field_validator("observation_time")
    @classmethod
    def end_of_hour(cls, value: datetime) -> datetime:
        if value.minute or value.second or value.microsecond:
            raise ValueError("observation_time must be an exact hourly interval end")
        return value


class TelemetryMessage(TelemetryFields):
    """One sensor observation. Sequence starts at 1 for each sensor/session pair."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "oneOf": [
                {"properties": {"sensor_id": {"const": sensor}, "station": {"const": station}}}
                for sensor, station in SENSOR_STATIONS.items()
            ]
        },
    )

    @model_validator(mode="after")
    def registered_mapping(self) -> TelemetryMessage:
        if SENSOR_STATIONS.get(self.sensor_id) != self.station:
            raise ValueError("unknown sensor or sensor-to-station mapping mismatch")
        return self
