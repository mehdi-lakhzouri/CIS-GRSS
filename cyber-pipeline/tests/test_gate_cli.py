import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from typer.testing import CliRunner

from sensorlab.cli import app
from sensorlab.gate.repository import SQLiteGateRepository

runner = CliRunner()


def test_operator_session_management(gate_signer, tmp_path):
    registry = tmp_path / "trusted.json"
    registry.write_text(gate_signer.registry.model_dump_json())
    database = tmp_path / "state" / "gate.sqlite3"
    session_id = str(uuid4())
    args = [
        "gate",
        "authorize",
        session_id,
        "--expires-at",
        (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "--registry",
        str(registry),
        "--database",
        str(database),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["authorized_sensors"]) == 6
    assert runner.invoke(app, args).exit_code == 1
    result = runner.invoke(app, ["gate", "status", "--database", str(database)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["accepted"] == 0
    assert len(json.loads(result.stdout)["sensors"]) == 6
    result = runner.invoke(app, ["gate", "revoke-session", session_id, "--database", str(database)])
    assert result.exit_code == 0
    assert all(row["revoked"] for row in SQLiteGateRepository(database).status()["sensors"])
    result = runner.invoke(app, ["gate", "events", "--database", str(database)])
    assert result.exit_code == 0 and json.loads(result.stdout) == []


def test_http_cli_requires_explicit_signed_authorized_session(release, tmp_path):
    output = tmp_path / "not-created.jsonl"
    options = ["replay", "--release", str(release), "--output", str(output)]
    assert (
        runner.invoke(app, [*options, "--endpoint", "https://localhost/v1/telemetry"]).exit_code
        != 0
    )
    assert runner.invoke(app, [*options, "--ca-cert", "unknown.pem"]).exit_code != 0
    assert not output.exists()


def test_replay_can_use_preauthorized_session_without_changing_unsigned_format(release):
    session_id = str(uuid4())
    result = runner.invoke(
        app, ["replay", "--release", str(release), "--speed", "0", "--session-id", session_id]
    )
    assert result.exit_code == 0
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(messages) == 18
    assert all(message["session_id"] == session_id for message in messages)
