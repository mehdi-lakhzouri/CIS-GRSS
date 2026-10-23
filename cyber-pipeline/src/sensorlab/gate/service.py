"""Ordered Crypto Gate policy pipeline, reusing the existing Ed25519 verifier."""

import hashlib
import re
from contextlib import suppress
from datetime import datetime
from uuid import UUID

from sensorlab.clock import Clock, SystemClock
from sensorlab.gate.models import MAX_SEQUENCE, Decision, GateConfig, IngressEnvelope, Packet
from sensorlab.gate.repository import GateRepository
from sensorlab.identity import TrustedRegistry
from sensorlab.signing import SignedEnvelope, verify_envelope
from sensorlab.wire import SecurityError, strict_json


class CryptoGate:
    def __init__(self, config: GateConfig, repository: GateRepository, clock: Clock | None = None):
        self.config = config
        self.repository = repository
        self.clock = clock or SystemClock()

    def handle(self, body: bytes, received_at: datetime) -> Decision:
        packet = Packet(timestamp=received_at, packet_digest=hashlib.sha256(body).hexdigest())
        try:
            raw = strict_json(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self.repository.reject(packet, "E_SCHEMA", "invalid or ambiguous JSON", 400)
        # Metadata is explicitly UNTRUSTED until verification, bounded and never the raw payload.
        telemetry = raw.get("telemetry")
        if isinstance(telemetry, dict):
            updates = {}
            for name in ("sensor_id", "station"):
                value = telemetry.get(name)
                if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
                    updates[name] = value
            with suppress(ValueError):
                updates["session_id"] = str(UUID(str(telemetry.get("session_id"))))
            sequence = telemetry.get("sequence")
            if type(sequence) is int and 1 <= sequence <= MAX_SEQUENCE:
                updates["sequence"] = sequence
            packet = packet.model_copy(update=updates)
        try:
            envelope = IngressEnvelope.model_validate(raw)
        except (ValueError, TypeError):
            return self.repository.reject(packet, "E_SCHEMA", "invalid signed envelope schema", 400)
        # Load the trust anchor for every request: revocation takes effect on the next load.
        try:
            trusted = TrustedRegistry.load(self.config.registry)
        except SecurityError:
            return self.repository.reject(packet, "E_REGISTRY", "trusted registry unavailable", 503)
        entry = next(
            (e for e in trusted.sensors if e.sensor_id == envelope.telemetry.sensor_id), None
        )
        if (
            entry is None
            or entry.key_id != envelope.key_id
            or entry.station != envelope.telemetry.station
        ):
            return self.repository.reject(
                packet, "E_UNKNOWN_KEY", "untrusted sensor/station/key", 403
            )
        if entry.status != "active":
            return self.repository.reject(packet, "E_REVOKED_KEY", "sensor key is revoked", 403)
        try:
            signed = SignedEnvelope.model_validate(raw)
            message = verify_envelope(signed, trusted)
        except (ValueError, TypeError):
            return self.repository.reject(packet, "E_BAD_SIG", "Ed25519 verification failed", 403)
        # Authorization, sequence, freshness, audit and durable Stage 2 handoff are atomic.
        return self.repository.admit(message, signed.key_id, packet, self.config, self.clock)
