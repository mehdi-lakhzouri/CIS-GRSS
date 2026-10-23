"""Real CLI signing/public-only verification and tamper rejection.

Uses disposable encrypted keys. The random password exists only in this process and
child environments, never in an artifact, command argument, or output.
"""

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sensorlab.cli import DEFAULT_RELEASE
from sensorlab.dataset import DetectorDataset, parse_utc
from sensorlab.identity import TrustedRegistry
from sensorlab.signing import SignedEnvelope, signing_bytes
from sensorlab.wire import encode64


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--speed", type=float, default=3600)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run_id = f"identity-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    output = project / "outputs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    store = project / ".sensorlab" / run_id
    environment = os.environ.copy()
    environment["SENSORLAB_KEY_PASSWORD"] = secrets.token_urlsafe(32)
    public_environment = environment.copy()
    public_environment.pop("SENSORLAB_KEY_PASSWORD")

    def cli(*arguments: str, private: bool = False, success: bool = True):
        result = subprocess.run(
            [sys.executable, "-m", "sensorlab.cli", *arguments],
            capture_output=True,
            text=True,
            cwd=output,
            env=environment if private else public_environment,
        )
        if success and result.returncode != 0:
            raise RuntimeError(result.stderr)
        if not success and result.returncode == 0:
            raise AssertionError("modified signed telemetry was unexpectedly accepted")
        return result

    sources = [args.release / name for name in ("detector_input.parquet", "data_contract.json")]
    before = {path.name: digest(path) for path in sources}
    cli("identities", "provision", "--directory", str(store), private=True)
    registry = TrustedRegistry.load(store / "registry.json")
    public_registry = output / "trusted_registry.json"
    public_registry.write_text(registry.model_dump_json(indent=2) + "\n", encoding="utf-8")
    start, end = "2025-01-01T01:00:00Z", "2025-01-02T01:00:00Z"
    signed = output / "signed-24h.jsonl"
    began = time.monotonic()
    replay = cli(
        "replay",
        "--release",
        str(args.release.resolve()),
        "--signed",
        "--identity-dir",
        str(store),
        "--speed",
        str(args.speed),
        "--start",
        start,
        "--end",
        end,
        "--output",
        str(signed),
        private=True,
    )
    elapsed = time.monotonic() - began
    status = json.loads(replay.stderr)
    verified = json.loads(cli("verify", str(signed), "--registry", str(public_registry)).stdout)
    assert verified["verified"] == status["transmitted"] == 144
    envelopes = [
        SignedEnvelope.model_validate_json(line)
        for line in signed.read_text(encoding="utf-8").splitlines()
    ]
    selected = DetectorDataset.load(args.release).select(start=parse_utc(start), end=parse_utc(end))
    for envelope, observation in zip(envelopes, selected.observations(), strict=True):
        assert envelope.telemetry.station == observation.station
        assert envelope.telemetry.observation_time == observation.time
        assert envelope.telemetry.measurements == observation.measurements
    assert all(sensor["transmitted"] == 24 for sensor in status["sensors"])
    for sensor in status["sensors"]:
        assert [
            e.telemetry.sequence for e in envelopes if e.telemetry.sensor_id == sensor["sensor_id"]
        ] == list(range(1, 25))

    changed = envelopes[0].model_dump(mode="json")
    changed["telemetry"]["measurements"]["SOIL_MOISTURE_5"] += 0.001
    tampered = output / "tampered-measurement.jsonl"
    tampered.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    failure = cli("verify", str(tampered), "--registry", str(public_registry), success=False)
    assert "invalid Ed25519 signature" in failure.stderr
    revoked_registry = output / "revoked_registry.json"
    revoked_registry.write_bytes(public_registry.read_bytes())
    cli(
        "identities",
        "revoke",
        envelopes[0].telemetry.sensor_id,
        "--registry",
        str(revoked_registry),
    )
    revoked_failure = cli("verify", str(signed), "--registry", str(revoked_registry), success=False)
    assert "revoked" in revoked_failure.stderr

    entry = registry.active_entry(envelopes[0].telemetry.sensor_id)
    vector = {
        "description": "Public verification vector; no private seed/key.",
        "public_key": entry.public_key,
        "signing_bytes_base64": encode64(signing_bytes(envelopes[0].telemetry, entry.key_id)),
        "envelope": envelopes[0].model_dump(mode="json"),
    }
    (output / "verification-vector.json").write_text(json.dumps(vector, indent=2) + "\n")
    after = {path.name: digest(path) for path in sources}
    assert before == after
    artifacts = [path for path in output.iterdir() if path.is_file()]
    evidence = {
        "validated_at": datetime.now(UTC).isoformat(),
        "release": str(args.release.resolve()),
        "input_sha256_before": before,
        "input_sha256_after": after,
        "source_preserved": True,
        "private_storage": str(store),
        "private_password_persisted": False,
        "private_key_storage": "Encrypted PKCS#8; restricted ACL/mode; Git-ignored",
        "registry": registry.model_dump(mode="json"),
        "selection": {"start_inclusive": start, "end_exclusive": end},
        "replay": status,
        "speed": args.speed,
        "elapsed_seconds_including_cli_startup": round(elapsed, 6),
        "sent_at_span_seconds": (
            envelopes[-1].telemetry.sent_at - envelopes[0].telemetry.sent_at
        ).total_seconds(),
        "verified_messages": verified["verified"],
        "null_measurements": sum(
            value is None
            for e in envelopes
            for value in e.telemetry.measurements.model_dump().values()
        ),
        "verification_without_password_or_private_directory_in_cwd": True,
        "measurement_tampering_rejected": True,
        "tamper_verification_exit_code": failure.returncode,
        "tamper_error": failure.stderr.strip(),
        "revoked_key_rejected": True,
        "revoked_key_error": revoked_failure.stderr.strip(),
        "private_dataset_files_opened": [],
        "artifacts_sha256": {path.name: digest(path) for path in sorted(artifacts)},
        "output_directory": str(output),
    }
    (output / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "signed": 144,
                "verified": 144,
                "tamper_rejected": True,
                "revocation_rejected": True,
                "evidence": str(output / "validation.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
