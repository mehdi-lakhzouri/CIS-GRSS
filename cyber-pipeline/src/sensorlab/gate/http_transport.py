"""HTTPS-only signed envelope sink. Certificate checks and redirects cannot be disabled."""

import math
import ssl
from pathlib import Path

import httpx

from sensorlab.signing import EnvelopeSink, SignedEnvelope
from sensorlab.wire import SecurityError


class HTTPSink:
    def __init__(self, endpoint: str, ca_cert: Path | None = None, timeout: float = 10):
        url = httpx.URL(endpoint)
        if (
            url.scheme != "https"
            or not url.host
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path != "/v1/telemetry"
        ):
            raise ValueError(
                "endpoint must be https://HOST[:PORT]/v1/telemetry without credentials/query"
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("HTTP timeout must be finite and positive")
        context = ssl.create_default_context(cafile=str(ca_cert) if ca_cert else None)
        self.client = httpx.Client(
            verify=context, timeout=timeout, trust_env=False, follow_redirects=False
        )
        self.endpoint = endpoint
        self.accepted = 0

    def write(self, message: SignedEnvelope) -> None:
        try:
            response = self.client.post(
                self.endpoint,
                content=message.model_dump_json(),
                headers={"content-type": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise SecurityError("HTTPS telemetry submission failed; acceptance is unknown") from exc
        if response.status_code != 202:
            # Controlled error category only; never echo remote payloads or raw request data.
            try:
                code = response.json().get("code", "E_REMOTE")
            except (ValueError, AttributeError):
                code = "E_REMOTE"
            if code not in {
                "E_SCHEMA",
                "E_UNKNOWN_KEY",
                "E_BAD_SIG",
                "E_UNAUTHORIZED_SESSION",
                "E_REPLAY_SEQ",
                "E_STALE",
                "E_REVOKED_KEY",
                "E_REGISTRY",
                "E_STORAGE",
                "E_PAYLOAD_TOO_LARGE",
                "E_HTTPS_REQUIRED",
            }:
                code = "E_REMOTE"
            raise SecurityError(f"Crypto Gate rejected telemetry ({code})")
        try:
            if response.json().get("decision") != "ACCEPT":
                raise SecurityError("invalid Crypto Gate acknowledgment")
        except (ValueError, AttributeError) as exc:
            raise SecurityError("invalid Crypto Gate acknowledgment") from exc
        self.accepted += 1

    def close(self) -> None:
        self.client.close()


class EnvelopeFanout:
    def __init__(self, *sinks: EnvelopeSink):
        self.sinks = sinks

    def write(self, message: SignedEnvelope) -> None:
        for sink in self.sinks:
            sink.write(message)
