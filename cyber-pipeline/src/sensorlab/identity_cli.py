"""Operator-controlled local provisioning and public registry administration."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from sensorlab.identity import DEFAULT_IDENTITY_DIR, TrustedRegistry, provision, revoke
from sensorlab.wire import SecurityError

identity_app = typer.Typer(
    help="Provision local identities and manage the trusted public registry."
)
DEFAULT_PASSWORD_ENV = "SENSORLAB_KEY_PASSWORD"


def key_password(environment: str, confirm: bool = False) -> bytes:
    value = os.environ.get(environment)
    if value is None:
        import sys

        if not sys.stdin.isatty():
            raise SecurityError(
                f"set {environment} or use a terminal for the hidden password prompt"
            )
        value = typer.prompt("Private-key password", hide_input=True, confirmation_prompt=confirm)
    if len(value.encode("utf-8")) < 16:
        raise SecurityError("key password must contain at least 16 UTF-8 bytes")
    return value.encode("utf-8")


@identity_app.command("provision")
def provision_command(
    directory: Annotated[
        Path, typer.Option(help="New, Git-ignored identity directory.")
    ] = DEFAULT_IDENTITY_DIR,
    password_env: Annotated[
        str, typer.Option(help="Environment variable; never pass a password.")
    ] = DEFAULT_PASSWORD_ENV,
) -> None:
    """Generate six independent encrypted Ed25519 key pairs; refuse existing storage."""
    try:
        registry = provision(directory, key_password(password_env, confirm=True))
    except (OSError, ValueError) as exc:
        typer.echo(f"Provisioning failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            {
                "registry": str(directory / "registry.json"),
                "sensors": len(registry.sensors),
                "algorithm": "Ed25519",
            }
        )
    )


@identity_app.command("list")
def list_identities(
    registry: Annotated[
        Path, typer.Option(help="Operator-trusted public registry.")
    ] = DEFAULT_IDENTITY_DIR / "registry.json",
) -> None:
    try:
        entries = TrustedRegistry.load(registry)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(entries.model_dump_json(indent=2))


@identity_app.command("revoke")
def revoke_command(
    sensor_id: str,
    registry: Annotated[
        Path, typer.Option(help="Registry to update atomically.")
    ] = DEFAULT_IDENTITY_DIR / "registry.json",
) -> None:
    """Mark a sensor's current key revoked for subsequent registry loads."""
    try:
        updated = revoke(registry, sensor_id)
    except (OSError, ValueError) as exc:
        typer.echo(f"Revocation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            {
                "sensor_id": sensor_id,
                "status": next(
                    entry.status for entry in updated.sensors if entry.sensor_id == sensor_id
                ),
            }
        )
    )
