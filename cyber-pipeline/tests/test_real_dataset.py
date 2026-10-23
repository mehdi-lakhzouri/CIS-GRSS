import hashlib
import io

import pandas as pd
import pytest

from sensorlab.dataset import DetectorDataset, parse_utc
from sensorlab.playback import PlaybackEngine
from sensorlab.registry import MEASUREMENT_COLUMNS, STATION_SENSORS
from sensorlab.telemetry import TelemetryMessage
from sensorlab.transport import NDJSONSink


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.real_dataset
def test_real_combined_24_hours_all_sensors(real_release, clock):
    paths = [real_release / name for name in ("detector_input.parquet", "data_contract.json")]
    before = {path.name: digest(path) for path in paths}
    assert before["detector_input.parquet"] == (
        "ff478b38f2f0996b5301a48f93f0a4b0a40b7bc321440c69f8e85ff31c6eb50a"
    )
    dataset = DetectorDataset.load(real_release)
    assert dataset.summary()["rows"] == 210384
    assert len(dataset.summary()["measurement_columns"]) == 19
    start, end = parse_utc("2025-01-01T01:00:00Z"), parse_utc("2025-01-02T01:00:00Z")
    selected = dataset.select(start=start, end=end)
    stream = io.StringIO()
    engine = PlaybackEngine(selected, NDJSONSink(stream), speed=3600, clock=clock)
    status = engine.run()
    assert status["state"] == "completed"
    assert status["transmitted"] == 144
    assert clock.elapsed == pytest.approx(23)
    messages = [
        TelemetryMessage.model_validate_json(line) for line in stream.getvalue().splitlines()
    ]
    assert {message.station for message in messages} == set(STATION_SENSORS)
    for station in STATION_SENSORS:
        assert [m.sequence for m in messages if m.station == station] == list(range(1, 25))
    # Compare directly to the public Parquet, not to any ground truth or audit data.
    frame = pd.read_parquet(paths[0])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame[(frame.time >= start) & (frame.time < end)].set_index(["time", "station"])
    for message in messages:
        expected = frame.loc[(message.observation_time, message.station)]
        for name in MEASUREMENT_COLUMNS:
            value = getattr(message.measurements, name)
            assert value is None if pd.isna(expected[name]) else value == expected[name]
    assert {path.name: digest(path) for path in paths} == before


@pytest.mark.real_dataset
def test_real_full_release_missingness(real_release):
    dataset = DetectorDataset.load(real_release)
    summary = dataset.summary()
    assert summary["missing_measurements"]["SOIL_MOISTURE_5"] == 5401
    assert summary["missing_measurements"]["SOIL_MOISTURE_10"] == 5335
    assert summary["missing_measurements"]["P_CALC"] == 1446
    # Stream every observation through the public models, including all missing cells.
    counts = dict.fromkeys(MEASUREMENT_COLUMNS, 0)
    total = 0
    for observation in dataset.observations():
        total += 1
        for name, value in observation.measurements.model_dump().items():
            counts[name] += value is None
    assert total == 210384
    assert counts == summary["missing_measurements"]
