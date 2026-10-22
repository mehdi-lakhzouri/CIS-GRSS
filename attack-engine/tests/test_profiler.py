from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from attack_engine.dataset.profiler import DatasetProfiler, ProfilingError
from attack_engine.dataset.schema import DatasetSchema
from attack_engine.dataset.validator import DatasetValidator


@pytest.fixture
def profiler(schema: DatasetSchema) -> DatasetProfiler:
    return DatasetProfiler(schema=schema)


def test_profile_reports_dimensions_period_and_stations(
    profiler: DatasetProfiler, observations: pd.DataFrame
) -> None:
    profile = profiler.profile(observations)

    assert profile.row_count == 96
    assert profile.column_count == 23
    assert profile.station_count == 2
    assert {station.row_count for station in profile.stations.values()} == {48}
    period = profile.observation_period
    assert period is not None
    assert period.first_observation_end.isoformat() == "2024-03-01T01:00:00+00:00"
    assert period.first_interval_start.isoformat() == "2024-03-01T00:00:00+00:00"
    assert period.last_observation_end.isoformat() == "2024-03-03T00:00:00+00:00"


def test_target_statistics_match_numpy(
    profiler: DatasetProfiler, observations: pd.DataFrame
) -> None:
    frame = observations.copy()
    frame.loc[[0, 1, 50], "SOIL_MOISTURE_5"] = np.nan
    expected = frame["SOIL_MOISTURE_5"].dropna().to_numpy()

    target = profiler.profile(frame).target

    assert target.missing_count == 3
    assert target.finite_count == 93
    assert target.coverage_percent == pytest.approx(100 * 93 / 96)
    assert target.minimum == expected.min()
    assert target.maximum == expected.max()
    assert target.mean == pytest.approx(expected.mean())
    assert target.std == pytest.approx(expected.std(ddof=1))
    assert target.unit == "m3/m3"


def test_missing_values_per_column_and_station(
    profiler: DatasetProfiler, observations: pd.DataFrame
) -> None:
    frame = observations.copy()
    first, second = frame["station"].unique()
    frame.loc[0, "SOIL_MOISTURE_5"] = np.nan
    frame.loc[[60, 61], "P_CALC"] = np.nan

    profile = profiler.profile(frame)

    assert profile.missing_by_column["SOIL_MOISTURE_5"] == 1
    assert profile.missing_by_column["P_CALC"] == 2
    assert profile.stations[first].missing_by_column == {"SOIL_MOISTURE_5": 1}
    assert profile.stations[second].missing_by_column == {"P_CALC": 2}
    assert profile.stations[first].target_missing_count == 1


def test_duplicates_gaps_and_year_boundary(profiler: DatasetProfiler, make_observations) -> None:
    frame = make_observations(start="2024-12-31 20:00", hours=8)
    frame = pd.concat([frame.drop(index=[2, 3]), frame.iloc[[0]]], ignore_index=True)

    profile = profiler.profile(frame)

    assert profile.duplicate_key_count == 1
    assert profile.temporal_gaps.gap_count == 1
    assert profile.temporal_gaps.missing_intervals == 2
    assert profile.temporal_gaps.largest[0].missing_intervals == 2
    assert profile.year_boundary_row_count == 2


def test_profile_includes_validation_summary(
    profiler: DatasetProfiler, validator: DatasetValidator, observations: pd.DataFrame
) -> None:
    frame = observations.assign(EXTRA=1)
    report = validator.validate(frame)

    summary = profiler.profile(frame, report).validation

    assert summary is not None
    assert summary.passed
    assert summary.warning_count == 1
    assert summary.warnings[0].startswith("unexpected_column")


def test_profile_is_json_serializable(
    profiler: DatasetProfiler, observations: pd.DataFrame
) -> None:
    document = json.loads(json.dumps(profiler.profile(observations).model_dump(mode="json")))
    assert document["target"]["column"] == "SOIL_MOISTURE_5"
    assert document["observation_period"]["first_observation_end"] == "2024-03-01T01:00:00Z"


def test_profile_does_not_modify_the_dataframe(
    profiler: DatasetProfiler, observations: pd.DataFrame
) -> None:
    snapshot = observations.copy(deep=True)
    profiler.profile(observations)
    pd.testing.assert_frame_equal(observations, snapshot, check_exact=True)


def test_profile_requires_key_columns(
    profiler: DatasetProfiler, observations: pd.DataFrame
) -> None:
    with pytest.raises(ProfilingError, match="station"):
        profiler.profile(observations.drop(columns=["station"]))
