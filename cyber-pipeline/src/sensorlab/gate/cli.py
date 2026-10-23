"""Local operator commands. No public management endpoints or automatic session trust."""

import json
import sqlite3
from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer
import uvicorn

from sensorlab.clock import SystemClock
from sensorlab.dataset import parse_utc
from sensorlab.gate.api import create_app
from sensorlab.gate.models import DEFAULT_DATABASE, GateConfig
from sensorlab.gate.repository import SQLiteGateRepository
from sensorlab.gate.tls import development_certificates
from sensorlab.identity import DEFAULT_IDENTITY_DIR, TrustedRegistry
from sensorlab.identity_cli import key_password

gate_app = typer.Typer(help="Stage 1 HTTPS Crypto Gate and local operator session management.")
DatabaseOption = Annotated[Path, typer.Option(help="Persistent SQLite security-state database.")]
RegistryOption = Annotated[Path, typer.Option(help="Operator-trusted public registry.")]


@gate_app.command("dev-cert")
def dev_cert(
    directory: Annotated[
        Path, typer.Option(help="New restricted development TLS directory.")
    ] = Path(".sensorlab/tls"),
    password_env: str = "SENSORLAB_TLS_PASSWORD",
) -> None:
    """Create a dedicated seven-day localhost certificate and CA; no system trust changes."""
    try:
        development_certificates(directory, key_password(password_env, confirm=True))
    except (OSError, ValueError) as exc:
        typer.echo(f"Certificate generation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            {
                "ca_certificate": str(directory / "ca.pem"),
                "server_certificate": str(directory / "server.pem"),
            }
        )
    )


@gate_app.command("authorize")
def authorize(
    session_id: UUID,
    expires_at: Annotated[
        str, typer.Option(help="Exclusive expiration, ISO timestamp with timezone.")
    ],
    database: DatabaseOption = DEFAULT_DATABASE,
    registry: RegistryOption = DEFAULT_IDENTITY_DIR / "registry.json",
    sensor_id: Annotated[
        list[str] | None, typer.Option(help="Repeat; default is all six sensors.")
    ] = None,
    not_before: str | None = None,
) -> None:
    """Explicitly authorize a new simulation UUID and fixed sensor/key bindings locally."""
    try:
        trusted = TrustedRegistry.load(registry)
        selected = sensor_id if sensor_id is not None else [e.sensor_id for e in trusted.sensors]
        if len(selected) != len(set(selected)):
            raise ValueError("select distinct sensors")
        entries = [trusted.active_entry(sensor) for sensor in selected]
        now = SystemClock().now()
        SQLiteGateRepository(database).authorize(
            session_id,
            entries,
            now,
            parse_utc(not_before) if not_before else now,
            parse_utc(expires_at),
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Authorization failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            {
                "session_id": str(session_id),
                "authorized_sensors": selected,
                "expires_at": parse_utc(expires_at).isoformat(),
            }
        )
    )


@gate_app.command("revoke-session")
def revoke_session(session_id: UUID, database: DatabaseOption = DEFAULT_DATABASE) -> None:
    try:
        SQLiteGateRepository(database).revoke_session(session_id)
    except (OSError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Session revocation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps({"session_id": str(session_id), "revoked": True}))


@gate_app.command("status")
def status(database: DatabaseOption = DEFAULT_DATABASE) -> None:
    try:
        result = SQLiteGateRepository(database).status()
    except (OSError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Status unavailable: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2))


@gate_app.command("events")
def events(database: DatabaseOption = DEFAULT_DATABASE, limit: int = 100) -> None:
    """Inspect structured security decisions, never full measurements or signatures."""
    try:
        result = SQLiteGateRepository(database).events(limit)
    except (OSError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Events unavailable: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2))


@gate_app.command("serve")
def serve(
    database: DatabaseOption = DEFAULT_DATABASE,
    registry: RegistryOption = DEFAULT_IDENTITY_DIR / "registry.json",
    cert: Path = Path(".sensorlab/tls/server.pem"),
    key: Path = Path(".sensorlab/tls/server.private.pem"),
    host: str = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8443,
    max_payload_bytes: int = 16384,
    max_age_seconds: float = 120,
    future_skew_seconds: float = 5,
    password_env: str = "SENSORLAB_TLS_PASSWORD",
) -> None:
    """Serve HTTPS only; proxy headers and access logs are disabled."""
    try:
        if not cert.is_file() or not key.is_file():
            raise ValueError("TLS certificate and encrypted key are required")
        application = create_app(
            GateConfig(
                registry=registry,
                database=database,
                max_payload_bytes=max_payload_bytes,
                max_age_seconds=max_age_seconds,
                future_skew_seconds=future_skew_seconds,
            )
        )
        password = key_password(password_env).decode("utf-8")
        uvicorn.run(
            application,
            host=host,
            port=port,
            ssl_certfile=str(cert),
            ssl_keyfile=str(key),
            ssl_keyfile_password=password,
            proxy_headers=False,
            access_log=False,
            log_level="warning",
            limit_concurrency=100,
            timeout_keep_alive=5,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Crypto Gate startup failed: {exc}", err=True)
        raise typer.Exit(1) from exc
