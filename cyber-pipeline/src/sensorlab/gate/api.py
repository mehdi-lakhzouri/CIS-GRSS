"""Bounded HTTPS-only ingestion. No network session/registry administration endpoints."""

import hashlib
import json
import logging
import sqlite3

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from sensorlab.clock import Clock, SystemClock
from sensorlab.gate.models import GateConfig, Packet
from sensorlab.gate.repository import SQLiteGateRepository
from sensorlab.gate.service import CryptoGate


def create_app(config: GateConfig, clock: Clock | None = None) -> FastAPI:
    # Fail startup if the configured trust anchor or repository cannot be initialized.
    from sensorlab.identity import TrustedRegistry

    TrustedRegistry.load(config.registry)
    clock = clock or SystemClock()
    repository = SQLiteGateRepository(config.database)
    gate = CryptoGate(config, repository, clock)
    app = FastAPI(
        title="SensorLab Crypto Gate",
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.repository = repository
    app.state.gate = gate

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/telemetry")
    async def telemetry(request: Request) -> JSONResponse:
        digest = hashlib.sha256()
        received_at = clock.now()
        try:

            async def reject(code: str, reason: str, status: int) -> JSONResponse:
                packet = Packet(
                    timestamp=clock.now(),
                    packet_digest=digest.hexdigest(),
                    digest_scope="received-prefix",
                )
                result = await run_in_threadpool(repository.reject, packet, code, reason, status)
                return JSONResponse(result.model_dump(mode="json"), status_code=result.http_status)

            if request.url.scheme != "https":
                return await reject("E_HTTPS_REQUIRED", "HTTPS is required", 403)
            content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if (
                content_type != "application/json"
                or request.headers.get("content-encoding", "identity") != "identity"
            ):
                return await reject(
                    "E_SCHEMA", "only uncompressed application/json is supported", 415
                )
            body = bytearray()
            async for chunk in request.stream():
                # Keep/hash at most limit+1 bytes, even if Content-Length is absent or misleading.
                prefix = chunk[: config.max_payload_bytes + 1 - len(body)]
                body.extend(prefix)
                digest.update(prefix)
                if len(body) > config.max_payload_bytes:
                    return await reject("E_PAYLOAD_TOO_LARGE", "payload limit exceeded", 413)
            received_at = clock.now()
            result = await run_in_threadpool(gate.handle, bytes(body), received_at)
            return JSONResponse(result.model_dump(mode="json"), status_code=result.http_status)
        except ClientDisconnect:
            return await reject("E_SCHEMA", "incomplete request body", 400)
        except (sqlite3.Error, OSError):
            # A storage failure must never permit acceptance. Do not log payloads/tracebacks.
            logging.getLogger("sensorlab.gate").error(
                json.dumps(
                    {
                        "timestamp": received_at.isoformat(),
                        "decision": "REJECT",
                        "code": "E_STORAGE",
                        "reason": "security state unavailable",
                        "packet_digest": digest.hexdigest(),
                    }
                )
            )
            return JSONResponse(
                {"decision": "REJECT", "code": "E_STORAGE", "reason": "security state unavailable"},
                status_code=503,
            )

    return app
