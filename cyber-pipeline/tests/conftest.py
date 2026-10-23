import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from sensorlab.dataset import DetectorDataset
from sensorlab.registry import MEASUREMENT_COLUMNS, STATION_SENSORS


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0
        self.origin = datetime(2026, 10, 23, tzinfo=UTC)
        self.sleeps = []
        self.on_sleep = None

    def now(self):
        return self.origin + timedelta(seconds=self.elapsed)

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.elapsed += seconds
        if self.on_sleep:
            self.on_sleep()


class MemorySink:
    def __init__(self):
        self.messages = []
        self.on_write = None

    def write(self, message):
        self.messages.append(message)
        if self.on_write:
            self.on_write(message)


@pytest.fixture
def release(tmp_path):
    rows = []
    for hour in range(3):
        for index, station in enumerate(STATION_SENSORS):
            rows.append(
                {
                    "station": station,
                    "time": datetime(2025, 1, 1, hour + 1),
                    "partition": "test",
                    **{column: float(index + hour / 10) for column in MEASUREMENT_COLUMNS},
                }
            )
    frame = pd.DataFrame(rows).sample(frac=1, random_state=14)
    frame.loc[frame.index[0], "P_CALC"] = float("nan")
    frame.to_parquet(tmp_path / "detector_input.parquet", index=False)
    contract = {
        "schema_version": "2.0",
        "key_columns": ["station", "time"],
        "detector_input_columns": list(frame.columns),
        "ground_truth_columns": ["is_attacked", "clean_target"],
    }
    (tmp_path / "data_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    # These must never be opened, even when present beside the allowed inputs.
    for name in (
        "row_labels.parquet",
        "cell_labels.parquet",
        "ground_truth.parquet",
        "attack_manifest.csv",
    ):
        (tmp_path / name).write_text("DO NOT READ", encoding="utf-8")
    return tmp_path


@pytest.fixture
def dataset(release):
    return DetectorDataset.load(release)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def sink():
    return MemorySink()


@pytest.fixture
def real_release():
    # Only the public detector release is used. No evaluator files or engine imports.
    path = (
        Path(__file__).resolve().parents[2]
        / "attack-engine/datasets/generated/multivariable-20261023-seed14-acb26f07"
        / "combined/detector"
    )
    if not (path / "detector_input.parquet").is_file():
        pytest.skip("local combined detector release is unavailable")
    return path


@pytest.fixture(scope="module")
def gate_signer(tmp_path_factory):
    from sensorlab.identity import provision
    from sensorlab.signing import SigningService

    password = b"crypto-gate-synthetic-test-password"
    directory = tmp_path_factory.mktemp("gate-keys") / "identities"
    provision(directory, password)
    return SigningService.unlock(directory, password)


@pytest.fixture
def gate_environment(tmp_path, gate_signer, dataset, clock, sink):
    from fastapi.testclient import TestClient

    from sensorlab.gate.api import create_app
    from sensorlab.gate.models import GateConfig
    from sensorlab.playback import PlaybackEngine

    PlaybackEngine(dataset, sink, speed=0, clock=clock).run()
    message = sink.messages[0]
    registry = tmp_path / "trusted.json"
    registry.write_text(gate_signer.registry.model_dump_json())
    config = GateConfig(registry=registry, database=tmp_path / "state" / "gate.sqlite3")
    application = create_app(config, clock)
    repository = application.state.repository
    repository.authorize(
        message.session_id,
        list(gate_signer.registry.sensors),
        clock.now(),
        clock.now(),
        clock.now() + timedelta(hours=1),
    )
    client = TestClient(application, base_url="https://testserver")
    return config, client, repository, message, gate_signer
