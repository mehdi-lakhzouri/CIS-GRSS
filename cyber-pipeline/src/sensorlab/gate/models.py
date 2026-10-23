"""Public decisions, ingestion configuration and the structural validation boundary."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sensorlab.signing import SignedEnvelope
from sensorlab.telemetry import TelemetryFields

DEFAULT_DATABASE = Path(".sensorlab/gate/gate.sqlite3")
MAX_SEQUENCE = 2**63 - 1


class GateTelemetry(TelemetryFields):
    # Schema first; sensor/station binding is the next, mandatory trusted-identity step.
    sequence: int = Field(ge=1, le=MAX_SEQUENCE, strict=True)
    sensor_id: str = Field(strict=True, min_length=1, max_length=100)
    station: str = Field(strict=True, min_length=1, max_length=100)


class IngressEnvelope(SignedEnvelope):
    telemetry: GateTelemetry


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    registry: Path
    database: Path = DEFAULT_DATABASE
    max_payload_bytes: int = Field(default=16384, ge=1024, le=1048576)
    max_age_seconds: float = Field(default=120, ge=0)
    future_skew_seconds: float = Field(default=5, ge=0)


class Packet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timestamp: datetime
    packet_digest: str
    digest_scope: Literal["body", "received-prefix"] = "body"
    sensor_id: str | None = None
    station: str | None = None
    session_id: str | None = None
    sequence: int | None = None


class Decision(Packet):
    decision_id: int | None = None
    decision: Literal["ACCEPT", "REJECT"]
    code: str
    reason: str
    http_status: int = Field(exclude=True)
    warnings: tuple[str, ...] = ()
    sequence_gap: int = 0


def utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone-aware timestamps required")
    return value.astimezone(UTC).isoformat(timespec="microseconds")
