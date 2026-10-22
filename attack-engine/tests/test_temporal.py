"""Interval-end semantics, continuity, leap years, year boundaries and partitions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from attack_engine.dataset import temporal
from attack_engine.dataset.schema import (
    DatasetSchema,
    PartitionPlan,
    PartitionSpec,
    ValidationProfile,
)
from attack_engine.dataset.validator import DatasetValidator, Severity

ObservationFactory = Callable[..., pd.DataFrame]
HOUR = timedelta(hours=1)

PLAN = PartitionPlan(
    partitions=(
        PartitionSpec(name="train", start="2022-01-01T00:00:00Z", end="2024-01-01T00:00:00Z"),
        PartitionSpec(name="validation", start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z"),
        PartitionSpec(name="test", start="2025-01-01T00:00:00Z", end="2026-01-01T00:00:00Z"),
    )
)


def _utc(*stamps: str) -> pd.Series:
    return temporal.to_utc(pd.Series(pd.to_datetime(list(stamps))))


# Timestamp interpretation ---------------------------------------------------------------


def test_naive_timestamps_are_interpreted_as_utc_without_mutation() -> None:
    source = pd.Series(pd.to_datetime(["2024-06-01 12:00"]))
    converted = temporal.to_utc(source)

    assert str(converted.dt.tz) == "UTC"
    assert converted.iloc[0] == pd.Timestamp("2024-06-01 12:00", tz="UTC")
    assert source.dt.tz is None


def test_aware_timestamps_are_converted_to_utc() -> None:
    source = pd.Series(pd.to_datetime(["2024-06-01 12:00"]).tz_localize("America/Chicago"))
    assert temporal.to_utc(source).iloc[0] == pd.Timestamp("2024-06-01 17:00", tz="UTC")


def test_unparseable_timestamps_are_reported(
    validator: DatasetValidator, make_observations: ObservationFactory
) -> None:
    frame = make_observations(hours=4)
    frame["time"] = frame["time"].astype(str).astype(object)
    frame.loc[1, "time"] = "not a timestamp"

    report = validator.validate(frame)

    assert {"incompatible_type", "unparseable_timestamp"} <= report.codes(Severity.ERROR)


def test_interval_start_and_observation_year() -> None:
    times = _utc("2024-01-01 00:00", "2024-01-01 01:00")
    assert list(temporal.interval_start(times, HOUR)) == list(
        _utc("2023-12-31 23:00", "2024-01-01 00:00")
    )
    assert list(temporal.observation_year(times, HOUR)) == [2023, 2024]


def test_unordered_rows_are_reported_not_reordered(
    validator: DatasetValidator, make_observations: ObservationFactory
) -> None:
    frame = make_observations().sample(frac=1.0, random_state=7)
    report = validator.validate(frame)

    assert report.passed
    assert "unordered_timestamps" in report.codes(Severity.WARNING)


# Hourly continuity ----------------------------------------------------------------------


def test_gap_in_hourly_series_is_an_error(
    validator: DatasetValidator, make_observations: ObservationFactory
) -> None:
    frame = make_observations().drop(index=[10, 11, 12])

    report = validator.validate(frame)

    gap = next(issue for issue in report.errors if issue.code == "temporal_gap")
    assert gap.count == 1
    assert "3 missing intervals" in gap.message


def test_gaps_can_be_downgraded_by_profile(
    schema: DatasetSchema, make_observations: ObservationFactory
) -> None:
    profile = ValidationProfile(name="gappy", allow_temporal_gaps=True)
    report = DatasetValidator(schema, profile).validate(make_observations().drop(index=[10]))

    assert report.passed
    assert "temporal_gap" in report.codes(Severity.WARNING)


def test_sub_hourly_frequency_is_rejected(
    validator: DatasetValidator, make_observations: ObservationFactory
) -> None:
    frame = make_observations(hours=6, stations=("AZ_Alpha_1_N",))
    frame["time"] = pd.date_range("2024-03-01 01:00", periods=6, freq="30min", unit="us")
    frame["annee"] = 2024

    codes = validator.validate(frame).codes(Severity.ERROR)

    assert {"misaligned_timestamp", "irregular_time_step"} <= codes


def test_analyze_steps_counts_missing_intervals() -> None:
    stations = pd.Series(["A_X", "A_X", "A_X", "B_Y", "B_Y"])
    times = _utc(
        "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 05:00",
        "2024-01-01 01:00", "2024-01-01 02:00",
    )  # fmt: skip

    analysis = temporal.analyze_steps(stations, times, HOUR)

    assert [(gap.station, gap.missing_intervals) for gap in analysis.gaps] == [("A_X", 2)]
    assert analysis.irregular_step_count == 0
    assert analysis.misaligned_count == 0


# Leap years ------------------------------------------------------------------------------


@pytest.mark.parametrize(("year", "hours"), [(2023, 8760), (2024, 8784), (2025, 8760)])
def test_intervals_in_year(year: int, hours: int) -> None:
    assert temporal.intervals_in_year(year, HOUR) == hours


def test_february_29_is_continuous(
    validator: DatasetValidator, make_observations: ObservationFactory
) -> None:
    frame = make_observations(start="2024-02-28 01:00", hours=72)

    report = validator.validate(frame)

    assert report.passed, report.errors
    leap_day = frame["time"].dt.strftime("%m-%d") == "02-29"
    assert leap_day.sum() == 24 * frame["station"].nunique()


def test_complete_leap_year_has_8784_records(
    schema: DatasetSchema, make_observations: ObservationFactory
) -> None:
    profile = ValidationProfile(name="complete", require_complete_years=True)
    validator = DatasetValidator(schema, profile)
    full_year = make_observations(start="2024-01-01 01:00", hours=8784, stations=("AZ_Alpha_1_N",))

    assert validator.validate(full_year).passed
    short = validator.validate(full_year.iloc[:-1])
    assert "incomplete_observation_year" in short.codes(Severity.ERROR)


def test_hours_spilling_into_next_year_are_detected(
    schema: DatasetSchema, make_observations: ObservationFactory
) -> None:
    frame = make_observations(start="2023-01-01 01:00", hours=8784, stations=("AZ_Alpha_1_N",))
    # Relabel the 24 spill-over hours of 2024 as 2023 to simulate a wrong year cut.
    frame["annee"] = 2023
    report = DatasetValidator(schema, ValidationProfile(name="p")).validate(frame)

    assert "observation_year_mismatch" in report.codes(Severity.ERROR)


# Year boundary ---------------------------------------------------------------------------


def test_midnight_january_first_belongs_to_previous_year(
    validator: DatasetValidator, make_observations: ObservationFactory
) -> None:
    frame = make_observations(start="2024-12-31 22:00", hours=4)

    report = validator.validate(frame)

    assert report.passed, report.errors
    midnight = frame["time"] == pd.Timestamp("2025-01-01 00:00")
    assert set(frame.loc[midnight, "annee"]) == {2024}
    boundary = next(issue for issue in report.issues if issue.code == "year_boundary_rows")
    assert boundary.count == frame["station"].nunique()


def test_calendar_year_labelling_is_rejected(
    validator: DatasetValidator, make_observations: ObservationFactory
) -> None:
    frame = make_observations(start="2024-12-31 22:00", hours=4)
    frame["annee"] = frame["time"].dt.year  # wrong: calendar year of the interval end

    report = validator.validate(frame)

    mismatch = next(issue for issue in report.errors if issue.code == "observation_year_mismatch")
    assert mismatch.count == frame["station"].nunique()


# Partitions ------------------------------------------------------------------------------


def test_partition_boundaries_follow_interval_end_semantics() -> None:
    times = _utc(
        "2022-01-01 01:00",  # first hour of 2022
        "2024-01-01 00:00",  # last hour of 2023
        "2024-01-01 01:00",  # first hour of 2024
        "2025-01-01 00:00",  # last hour of 2024
        "2026-01-01 00:00",  # last hour of 2025
        "2022-01-01 00:00",  # last hour of 2021: outside every partition
    )

    labels = temporal.assign_partitions(times, PLAN, HOUR)

    assert list(labels) == ["train", "train", "validation", "validation", "test", None]


def test_every_record_is_in_exactly_one_partition(
    schema: DatasetSchema, make_observations: ObservationFactory
) -> None:
    frame = make_observations(start="2023-12-31 23:00", hours=4)
    validator = DatasetValidator(schema, ValidationProfile(name="p"), partitions=PLAN)

    report = validator.validate(frame)

    assert report.passed, report.errors
    assert report.partition_row_counts == {"train": 4, "validation": 4, "test": 0}
    assert sum(report.partition_row_counts.values()) == len(frame)


def test_records_outside_partitions_are_errors(
    schema: DatasetSchema, make_observations: ObservationFactory
) -> None:
    frame = make_observations(start="2021-12-31 23:00", hours=3)
    validator = DatasetValidator(schema, ValidationProfile(name="p"), partitions=PLAN)

    unassigned = next(
        issue for issue in validator.validate(frame).errors if issue.code == "unassigned_partition"
    )
    assert unassigned.count == 2 * frame["station"].nunique()


def test_overlapping_partitions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        PartitionPlan(
            partitions=(
                PartitionSpec(name="a", start="2022-01-01T00:00:00Z", end="2024-01-01T01:00:00Z"),
                PartitionSpec(name="b", start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z"),
            )
        )


def test_adjacent_partitions_do_not_overlap() -> None:
    assert len(PLAN.partitions) == 3


@pytest.mark.parametrize(
    ("start", "end"),
    [("2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"), ("2024-01-01T00:00:00", "2025-01-01")],
)
def test_invalid_partition_bounds(start: str, end: str) -> None:
    with pytest.raises(ValidationError):
        PartitionSpec(name="bad", start=start, end=end)


def test_duplicate_partition_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        PartitionPlan(
            partitions=(
                PartitionSpec(name="a", start="2022-01-01T00:00:00Z", end="2023-01-01T00:00:00Z"),
                PartitionSpec(name="a", start="2023-01-01T00:00:00Z", end="2024-01-01T00:00:00Z"),
            )
        )
