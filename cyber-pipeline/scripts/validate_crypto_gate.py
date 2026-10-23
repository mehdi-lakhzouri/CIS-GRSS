"""Real CLI -> signing -> verified HTTPS -> persistent Crypto Gate; no Stage 2 consumer.

Disposable passwords exist only in process environments. Stops only owned server processes.
"""

import hashlib
import json
import os
import secrets
import socket
import ssl
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx

from sensorlab.cli import DEFAULT_RELEASE
from sensorlab.dataset import DetectorDataset, parse_utc
from sensorlab.gate.repository import SQLiteGateRepository
from sensorlab.identity import protect_directory
from sensorlab.signing import SignedEnvelope, SigningService


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    run_id = f"gate-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    output, security = project / "outputs" / run_id, project / ".sensorlab" / run_id
    output.mkdir(parents=True, exist_ok=False)
    security.mkdir(mode=0o700, parents=True, exist_ok=False)
    protect_directory(security)
    identity, tls, database = (
        security / "identity",
        security / "tls",
        security / "state/gate.sqlite3",
    )
    sensor_password, tls_password = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    signing_env = os.environ.copy()
    signing_env["SENSORLAB_KEY_PASSWORD"] = sensor_password
    tls_env = os.environ.copy()
    tls_env.pop("SENSORLAB_KEY_PASSWORD", None)
    tls_env["SENSORLAB_TLS_PASSWORD"] = tls_password
    public_env = os.environ.copy()
    for name in ("SENSORLAB_KEY_PASSWORD", "SENSORLAB_TLS_PASSWORD"):
        public_env.pop(name, None)

    def cli(*arguments: str, environment=public_env):
        result = subprocess.run(
            [sys.executable, "-m", "sensorlab.cli", *arguments],
            capture_output=True,
            text=True,
            cwd=project,
            env=environment,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result

    sources = [DEFAULT_RELEASE / name for name in ("detector_input.parquet", "data_contract.json")]
    before = {path.name: sha256(path) for path in sources}
    cli("identities", "provision", "--directory", str(identity), environment=signing_env)
    cli("gate", "dev-cert", "--directory", str(tls), environment=tls_env)
    registry = output / "trusted_registry.json"
    registry.write_bytes((identity / "registry.json").read_bytes())
    session_id = uuid4()
    cli(
        "gate",
        "authorize",
        str(session_id),
        "--expires-at",
        (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "--database",
        str(database),
        "--registry",
        str(registry),
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"https://127.0.0.1:{port}"
    context = ssl.create_default_context(cafile=str(tls / "ca.pem"))
    server = None
    log = (output / "server.log").open("x", encoding="utf-8")

    def start_server():
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sensorlab.cli",
                "gate",
                "serve",
                "--port",
                str(port),
                "--registry",
                str(registry),
                "--database",
                str(database),
                "--cert",
                str(tls / "server.pem"),
                "--key",
                str(tls / "server.private.pem"),
            ],
            cwd=project,
            env=tls_env,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + 15
            with httpx.Client(verify=context, trust_env=False, timeout=1) as client:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Crypto Gate subprocess stopped during startup")
                    try:
                        if client.get(base + "/health").status_code == 200:
                            return process
                    except httpx.HTTPError:
                        time.sleep(0.05)
            raise RuntimeError("Crypto Gate HTTPS startup timeout")
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise

    try:
        server = start_server()
        signed = output / "signed-https-24h.jsonl"
        start, end = "2025-01-01T01:00:00Z", "2025-01-02T01:00:00Z"
        began = time.monotonic()
        replay = cli(
            "replay",
            "--signed",
            "--identity-dir",
            str(identity),
            "--session-id",
            str(session_id),
            "--speed",
            "3600",
            "--start",
            start,
            "--end",
            end,
            "--endpoint",
            base + "/v1/telemetry",
            "--ca-cert",
            str(tls / "ca.pem"),
            "--output",
            str(signed),
            environment=signing_env,
        )
        elapsed = time.monotonic() - began
        replay_status = json.loads(replay.stderr)
        repository = SQLiteGateRepository(database)
        baseline = repository.status()
        assert (
            baseline["accepted"] == baseline["outbox_count"] == replay_status["transmitted"] == 144
        )
        assert baseline["rejected"] == baseline["sequence_gap_events"] == 0
        assert all(sensor["last_sequence"] == 24 for sensor in baseline["sensors"])
        envelopes = [
            SignedEnvelope.model_validate_json(line)
            for line in signed.read_text(encoding="utf-8").splitlines()
        ]
        observations = DetectorDataset.load(DEFAULT_RELEASE).select(
            start=parse_utc(start), end=parse_utc(end)
        )
        for envelope, observation in zip(envelopes, observations.observations(), strict=True):
            assert envelope.telemetry.measurements == observation.measurements
            assert envelope.telemetry.observation_time == observation.time
        signer = SigningService.unlock(identity, sensor_password.encode())
        cases = {}
        with httpx.Client(verify=context, trust_env=False, timeout=10) as client:

            def send(name, packet, expected):
                response = client.post(base + "/v1/telemetry", json=packet)
                body = response.json()
                assert body["code"] == expected, (name, body)
                cases[name] = body
                return body

            send("duplicate", envelopes[0].model_dump(mode="json"), "E_REPLAY_SEQ")
            tampered = envelopes[0].model_dump(mode="json")
            tampered["telemetry"]["measurements"]["SOIL_MOISTURE_5"] += 0.001
            send("tampered_measurement", tampered, "E_BAD_SIG")
            message = next(
                e.telemetry
                for e in reversed(envelopes)
                if e.telemetry.sensor_id == envelopes[0].telemetry.sensor_id
            )

            def fresh(**updates):
                return signer.sign(
                    message.model_copy(update={"sent_at": datetime.now(UTC), **updates})
                )

            gap = send("gap", fresh(sequence=26).model_dump(mode="json"), "OK")
            assert gap["warnings"] == ["W_SEQ_GAP"] and gap["sequence_gap"] == 1
            send("out_of_order", fresh(sequence=25).model_dump(mode="json"), "E_REPLAY_SEQ")
            send(
                "unauthorized_session",
                fresh(sequence=1, session_id=uuid4()).model_dump(mode="json"),
                "E_UNAUTHORIZED_SESSION",
            )
            send(
                "stale",
                fresh(sequence=27, sent_at=datetime.now(UTC) - timedelta(minutes=5)).model_dump(
                    mode="json"
                ),
                "E_STALE",
            )
            send(
                "future",
                fresh(sequence=27, sent_at=datetime.now(UTC) + timedelta(minutes=5)).model_dump(
                    mode="json"
                ),
                "E_STALE",
            )
            wrong_key = fresh(sequence=27).model_dump(mode="json")
            wrong_key["key_id"] = signer.registry.sensors[1].key_id
            send("wrong_key", wrong_key, "E_UNKNOWN_KEY")
            server.terminate()
            server.wait(timeout=10)
            server = start_server()
            send("replay_after_restart", fresh(sequence=26).model_dump(mode="json"), "E_REPLAY_SEQ")
            send("valid_after_restart", fresh(sequence=27).model_dump(mode="json"), "OK")
            cli("identities", "revoke", message.sensor_id, "--registry", str(registry))
            send("revoked_key", fresh(sequence=28).model_dump(mode="json"), "E_REVOKED_KEY")
        final = repository.status()
        assert final["accepted"] == final["outbox_count"] == 146
        assert final["rejected"] == 9
        assert final["sequence_gap_events"] == 1
        after = {path.name: sha256(path) for path in sources}
        assert before == after
        (output / "security-events.json").write_text(json.dumps(repository.events(1000), indent=2))
        evidence = {
            "validated_at": datetime.now(UTC).isoformat(),
            "release": str(DEFAULT_RELEASE),
            "input_sha256_before": before,
            "input_sha256_after": after,
            "source_preserved": True,
            "output_directory": str(output),
            "database": str(database),
            "session_id": str(session_id),
            "endpoint": base + "/v1/telemetry",
            "tls_certificate_verified": True,
            "test_passwords_persisted": False,
            "selection": {"start_inclusive": start, "end_exclusive": end},
            "replay": replay_status,
            "replay_elapsed_seconds": round(elapsed, 6),
            "initial_gate_accounting": baseline,
            "adversarial_decisions": cases,
            "final_gate_accounting": final,
            "restart_protection_verified": True,
            "stage2_outbox_rows": 146,
            "stage2_consumer_implemented": False,
            "artifacts_sha256": {
                path.name: sha256(path)
                for path in (
                    signed,
                    registry,
                    output / "security-events.json",
                    tls / "ca.pem",
                    tls / "server.pem",
                )
            },
        }
        (output / "validation.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "real_messages_accepted": 144,
                    "final_accepted": 146,
                    "rejected": 9,
                    "restart_protected": True,
                    "evidence": str(output / "validation.json"),
                }
            )
        )
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    main()
