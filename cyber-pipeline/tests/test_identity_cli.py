import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sensorlab.cli import app
from sensorlab.identity import provision
from sensorlab.signing import SignedEnvelope

PASSWORD = "cli-synthetic-password-not-production"
runner = CliRunner()


@pytest.fixture(scope="module")
def cli_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("identity-cli") / "keys"
    provision(root, PASSWORD.encode())
    return root


def test_provision_cli_secure_and_no_overwrite(tmp_path, monkeypatch):
    root = tmp_path / "new"
    monkeypatch.setenv("SENSORLAB_KEY_PASSWORD", PASSWORD)
    result = runner.invoke(app, ["identities", "provision", "--directory", str(root)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["sensors"] == 6
    assert PASSWORD not in result.output
    assert "PRIVATE KEY" not in result.output
    result = runner.invoke(app, ["identities", "provision", "--directory", str(root)])
    assert result.exit_code == 1


def test_signed_replay_verify_and_sign_existing(cli_store, release, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORLAB_KEY_PASSWORD", PASSWORD)
    unsigned, signed, offline = (tmp_path / name for name in ("u.jsonl", "s.jsonl", "o.jsonl"))
    args = ["replay", "--release", str(release), "--speed", "0"]
    result = runner.invoke(app, [*args, "--output", str(unsigned)])
    assert result.exit_code == 0
    result = runner.invoke(
        app, [*args, "--output", str(signed), "--signed", "--identity-dir", str(cli_store)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stderr)["transmitted"] == 18
    public_registry = tmp_path / "trusted.json"
    public_registry.write_bytes((cli_store / "registry.json").read_bytes())
    # Verification needs no password, private directory, or dataset.
    monkeypatch.delenv("SENSORLAB_KEY_PASSWORD")
    result = runner.invoke(app, ["verify", str(signed), "--registry", str(public_registry)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["verified"] == 18
    monkeypatch.setenv("SENSORLAB_KEY_PASSWORD", PASSWORD)
    result = runner.invoke(
        app, ["sign", str(unsigned), "--output", str(offline), "--identity-dir", str(cli_store)]
    )
    assert result.exit_code == 0, result.output
    messages = [json.loads(line) for line in unsigned.read_text().splitlines()]
    envelopes = [json.loads(line) for line in offline.read_text().splitlines()]
    assert [envelope["telemetry"] for envelope in envelopes] == messages
    before = offline.read_bytes()
    result = runner.invoke(
        app, ["sign", str(unsigned), "--output", str(offline), "--identity-dir", str(cli_store)]
    )
    assert result.exit_code == 1
    assert offline.read_bytes() == before
    altered = json.loads(signed.read_text().splitlines()[0])
    altered["telemetry"]["measurements"]["SOIL_MOISTURE_5"] += 0.01
    signed.write_text(json.dumps(altered) + "\n")
    result = runner.invoke(app, ["verify", str(signed), "--registry", str(public_registry)])
    assert result.exit_code == 1
    assert "invalid Ed25519 signature" in result.stderr


def test_signed_schema_and_inventory(cli_store):
    result = runner.invoke(app, ["schema", "--signed"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == SignedEnvelope.model_json_schema()
    schema = Path(__file__).resolve().parents[1] / "schemas/signed-telemetry-v1.schema.json"
    assert json.loads(schema.read_text()) == SignedEnvelope.model_json_schema()
    result = runner.invoke(
        app, ["identities", "list", "--registry", str(cli_store / "registry.json")]
    )
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["sensors"]) == 6


def test_revoke_cli_then_reject_signing(cli_store, tmp_path, release, monkeypatch):
    public = tmp_path / "revoked.json"
    public.write_bytes((cli_store / "registry.json").read_bytes())
    sensor_id = "uscrn-az-tucson-11-w"
    result = runner.invoke(app, ["identities", "revoke", sensor_id, "--registry", str(public)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "revoked"
    # Use an isolated public store to assert revoked status fails before a missing key lookup.
    root = tmp_path / "revoked-store"
    root.mkdir()
    (root / "registry.json").write_bytes(public.read_bytes())
    monkeypatch.setenv("SENSORLAB_KEY_PASSWORD", PASSWORD)
    output = tmp_path / "should-not-exist.jsonl"
    result = runner.invoke(
        app,
        [
            "replay",
            "--release",
            str(release),
            "--speed",
            "0",
            "--signed",
            "--station",
            "AZ_Tucson_11_W",
            "--identity-dir",
            str(root),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert "revoked" in result.stderr
    assert not output.exists()


def test_password_failure_does_not_write_output(cli_store, release, tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORLAB_KEY_PASSWORD", "incorrect-but-long-enough")
    output = tmp_path / "failed.jsonl"
    result = runner.invoke(
        app,
        [
            "replay",
            "--release",
            str(release),
            "--speed",
            "0",
            "--signed",
            "--identity-dir",
            str(cli_store),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert "incorrect-but-long-enough" not in result.output
    assert not output.exists()
