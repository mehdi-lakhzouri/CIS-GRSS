"""Versioned signed envelopes and public-registry-only, stateless verification."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from sensorlab.identity import KEY_PATTERN, RegistryEntry, TrustedRegistry, load_private_key
from sensorlab.telemetry import TelemetryMessage
from sensorlab.transport import MessageSigner
from sensorlab.wire import SecurityError, decode64, encode64, strict_json

DOMAIN = b"SENSORLAB-SIGNED-TELEMETRY\x00v1\x00"


class SignedEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    envelope_version: Literal["1.0"]
    algorithm: Literal["Ed25519"]
    serialization: Literal["sensorlab-c14n-1"]
    key_id: str = Field(strict=True, pattern=KEY_PATTERN)
    telemetry: TelemetryMessage
    signature: str = Field(strict=True)

    @field_validator("signature")
    @classmethod
    def signature_encoding(cls, value: str) -> str:
        decode64(value, 64)
        return value


def signing_bytes(telemetry: TelemetryMessage, key_id: str) -> bytes:
    """sensorlab-c14n-1: documented semantic encoding, not RFC 8785/JCS.

    No data is changed in the transmitted telemetry. Only the *signing representation*
    uses hexadecimal IEEE-754 binary64 strings for measurement numbers.
    """
    # Revalidate even caller-constructed models instead of trusting model_copy/construct.
    telemetry = TelemetryMessage.model_validate(telemetry.model_dump(mode="json"))
    values = telemetry.model_dump(mode="json")
    for name in ("observation_time", "sent_at"):
        timestamp: datetime = getattr(telemetry, name)
        values[name] = (
            timestamp.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        )
    values["measurements"] = {
        name: None if value is None else float(value).hex()
        for name, value in telemetry.measurements.model_dump().items()
    }
    body = {
        "envelope_version": "1.0",
        "algorithm": "Ed25519",
        "serialization": "sensorlab-c14n-1",
        "key_id": key_id,
        "telemetry": values,
    }
    return DOMAIN + json.dumps(
        body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class Ed25519Signer:
    """Implements MessageSigner using a private key kept out of its representation."""

    def __init__(self, key: Ed25519PrivateKey):
        self._key = key

    def sign(self, payload: bytes) -> bytes:
        return self._key.sign(payload)


class SigningService:
    def __init__(self, registry: TrustedRegistry, signers: dict[str, MessageSigner]):
        self.registry = registry
        self._signers = dict(signers)

    @classmethod
    def unlock(
        cls, directory: Path, password: bytes, sensor_ids: set[str] | None = None
    ) -> SigningService:
        registry = TrustedRegistry.load(directory / "registry.json")
        selected = (
            sensor_ids
            if sensor_ids is not None
            else {entry.sensor_id for entry in registry.sensors if entry.status == "active"}
        )
        signers = {}
        for sensor_id in sorted(selected):
            entry = registry.active_entry(sensor_id)
            signers[sensor_id] = Ed25519Signer(load_private_key(directory, entry, password))
        return cls(registry, signers)

    def sign(self, telemetry: TelemetryMessage) -> SignedEnvelope:
        entry = self.registry.active_entry(telemetry.sensor_id)
        signer = self._signers.get(telemetry.sensor_id)
        if signer is None:
            raise SecurityError("sensor signing key is not unlocked")
        signature = signer.sign(signing_bytes(telemetry, entry.key_id))
        envelope = SignedEnvelope(
            envelope_version="1.0",
            algorithm="Ed25519",
            serialization="sensorlab-c14n-1",
            key_id=entry.key_id,
            telemetry=telemetry,
            signature=encode64(signature),
        )
        # Detect mismatched custom adapters before emitting any incorrectly signed data.
        verify_envelope(envelope, self.registry)
        return envelope


def verify_envelope(envelope: SignedEnvelope, registry: TrustedRegistry) -> TelemetryMessage:
    """Requires only public trust data. No private storage, network, or replay ledger."""
    try:
        envelope = SignedEnvelope.model_validate(envelope.model_dump(mode="json"))
        entry: RegistryEntry = registry.active_entry(envelope.telemetry.sensor_id, envelope.key_id)
        public_key = Ed25519PublicKey.from_public_bytes(decode64(entry.public_key, 32))
        public_key.verify(
            decode64(envelope.signature, 64), signing_bytes(envelope.telemetry, envelope.key_id)
        )
    except InvalidSignature as exc:
        raise SecurityError("invalid Ed25519 signature") from exc
    except (ValidationError, TypeError) as exc:
        raise SecurityError("invalid signed telemetry envelope") from exc
    return envelope.telemetry


def verify_json(text: str, registry: TrustedRegistry) -> TelemetryMessage:
    try:
        envelope = SignedEnvelope.model_validate(strict_json(text))
    except (ValueError, TypeError) as exc:
        raise SecurityError("invalid signed telemetry envelope") from exc
    return verify_envelope(envelope, registry)


def parse_unsigned(text: str) -> TelemetryMessage:
    try:
        return TelemetryMessage.model_validate(strict_json(text))
    except (ValueError, TypeError) as exc:
        raise SecurityError("invalid unsigned telemetry message") from exc


class EnvelopeSink(Protocol):
    def write(self, message: SignedEnvelope) -> None: ...


class SignedTelemetrySink:
    """TelemetrySink adapter: playback still emits the original TelemetryMessage."""

    def __init__(self, service: SigningService, output: EnvelopeSink):
        self.service = service
        self.output = output

    def write(self, message: TelemetryMessage) -> None:
        self.output.write(self.service.sign(message))
