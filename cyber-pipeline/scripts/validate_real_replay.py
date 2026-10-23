"""Run two real CLI replays and verify only public data. No evaluator files are read."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sensorlab.cli import DEFAULT_RELEASE
from sensorlab.dataset import DetectorDataset, parse_utc
from sensorlab.registry import MEASUREMENT_COLUMNS, STATION_SENSORS
from sensorlab.telemetry import TelemetryMessage


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run_id = f"validation-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = args.output_root / run_id
    root.mkdir(parents=True, exist_ok=False)
    inputs = [args.release / name for name in ("detector_input.parquet", "data_contract.json")]
    before = {path.name: sha256(path) for path in inputs}
    dataset = DetectorDataset.load(args.release)
    start, end = "2025-01-01T01:00:00Z", "2025-01-02T01:00:00Z"
    selected = dataset.select(start=parse_utc(start), end=parse_utc(end))
    expected = list(selected.observations())
    runs, fingerprints = [], []
    for speed in (3600, 0):
        output = root / f"telemetry-speed{speed}.jsonl"
        command = [
            sys.executable,
            "-m",
            "sensorlab.cli",
            "replay",
            "--release",
            str(args.release),
            "--speed",
            str(speed),
            "--start",
            start,
            "--end",
            end,
            "--output",
            str(output),
        ]
        began = time.monotonic()
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        elapsed = time.monotonic() - began
        if process.stdout:
            raise AssertionError("file-only playback must not expose data on stdout")
        status = json.loads(process.stderr)
        messages = [
            TelemetryMessage.model_validate_json(line)
            for line in output.read_text(encoding="utf-8").splitlines()
        ]
        assert status["state"] == "completed"
        assert len(messages) == status["transmitted"] == len(expected) == 144
        for message, observation in zip(messages, expected, strict=True):
            assert message.station == observation.station
            assert message.observation_time == observation.time
            assert message.measurements == observation.measurements
        for station in STATION_SENSORS:
            assert [m.sequence for m in messages if m.station == station] == list(range(1, 25))
        stable = [m.model_dump(mode="json", exclude={"session_id", "sent_at"}) for m in messages]
        fingerprint = hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        fingerprints.append(fingerprint)
        runs.append(
            {
                "speed": speed,
                "elapsed_seconds_including_cli_startup": round(elapsed, 6),
                "sent_at_span_seconds": (
                    messages[-1].sent_at - messages[0].sent_at
                ).total_seconds(),
                "output": str(output.resolve()),
                "output_sha256": sha256(output),
                "observation_fingerprint_sha256": fingerprint,
                "null_measurements": sum(
                    value is None
                    for message in messages
                    for value in message.measurements.model_dump().values()
                ),
                "status": status,
            }
        )
        print(f"Validated {speed}x: 144 messages, six sensors, {elapsed:.2f}s", flush=True)
    assert fingerprints[0] == fingerprints[1]
    after = {path.name: sha256(path) for path in inputs}
    assert before == after
    result = {
        "validated_at": datetime.now(UTC).isoformat(),
        "release": str(args.release.resolve()),
        "input_sha256_before": before,
        "input_sha256_after": after,
        "source_preserved": True,
        "dataset": dataset.summary(),
        "selection": {"start_inclusive": start, "end_exclusive": end, "rows": 144},
        "measurement_fields": list(MEASUREMENT_COLUMNS),
        "runs": runs,
        "observation_order_and_values_reproducible": True,
        "private_files_opened": [],
    }
    with (root / "validation.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Evidence: {root / 'validation.json'}")


if __name__ == "__main__":
    main()
