import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from sensorlab.dataset import DatasetError, DetectorDataset, parse_utc
from sensorlab.registry import MEASUREMENT_COLUMNS, STATION_SENSORS


def test_exact_public_inputs_and_missing_values(release, monkeypatch):
    original_text, original_read = Path.read_text, pd.read_parquet
    accessed = []

    def text(path, *args, **kwargs):
        assert path.name == "data_contract.json"
        accessed.append(path.name)
        return original_text(path, *args, **kwargs)

    def parquet(path, *args, **kwargs):
        assert path.name == "detector_input.parquet"
        accessed.append(path.name)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", text)
    monkeypatch.setattr(pd, "read_parquet", parquet)
    dataset = DetectorDataset.load(release)
    assert accessed == ["data_contract.json", "detector_input.parquet"]
    observations = list(dataset.observations())
    assert len(observations) == 18
    assert sum(row.measurements.P_CALC is None for row in observations) == 1
    assert all(
        set(row.measurements.model_dump()) == set(MEASUREMENT_COLUMNS) for row in observations
    )


@pytest.mark.parametrize("column", ["is_attacked", "clean_target", "attack_id", "extra"])
@pytest.mark.parametrize("in_contract", [False, True])
def test_rejects_extra_or_label_columns(release, column, in_contract):
    path = release / "detector_input.parquet"
    frame = pd.read_parquet(path)
    frame[column] = 1
    frame.to_parquet(path, index=False)
    if in_contract:
        contract_path = release / "data_contract.json"
        contract = json.loads(contract_path.read_text())
        contract["detector_input_columns"].append(column)
        contract_path.write_text(json.dumps(contract))
    with pytest.raises((DatasetError, ValidationError)):
        DetectorDataset.load(release)


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate",
        "unknown_station",
        "missing_station",
        "time",
        "infinity",
        "text_measurement",
        "bool_measurement",
        "partition",
    ],
)
def test_invalid_release(release, problem):
    path = release / "detector_input.parquet"
    frame = pd.read_parquet(path)
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "unknown_station":
        frame.loc[0, "station"] = "foreign"
    elif problem == "missing_station":
        frame = frame[frame.station != next(iter(STATION_SENSORS))]
    elif problem == "time":
        frame.loc[0, "time"] += pd.Timedelta(minutes=1)
    elif problem == "infinity":
        frame.loc[0, "SOIL_MOISTURE_10"] = float("inf")
    elif problem == "text_measurement":
        frame["P_CALC"] = "rain"
    elif problem == "bool_measurement":
        frame["P_CALC"] = True
    else:
        frame.loc[0, "partition"] = "private"
    frame.to_parquet(path, index=False)
    with pytest.raises(DatasetError):
        DetectorDataset.load(release)


def test_selection_and_interval_end_semantics(dataset):
    station = next(iter(STATION_SENSORS))
    selected = dataset.select(
        [station], parse_utc("2025-01-01T01:00:00Z"), parse_utc("2025-01-01T03:00:00Z")
    )
    assert [row.time.hour for row in selected.observations()] == [1, 2]
    assert selected.summary()["stations"] == {station: 2}


@pytest.mark.parametrize("stations", [["unknown"], [], ["AZ_Tucson_11_W"] * 2])
def test_bad_station_selection(dataset, stations):
    with pytest.raises(DatasetError):
        dataset.select(stations)


def test_bad_date_selection(dataset):
    with pytest.raises(DatasetError):
        parse_utc("2025-01-01")
    with pytest.raises(DatasetError):
        dataset.select(start=datetime(2025, 1, 1))
    with pytest.raises(DatasetError):
        dataset.select(start=parse_utc("2030-01-01T00:00Z"))
    with pytest.raises(DatasetError):
        dataset.select(start=parse_utc("2025-01-02T00:00Z"), end=parse_utc("2025-01-01T00:00Z"))
