from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attack_engine.dataset.schema import DatasetSchema, ValidationProfile
from attack_engine.dataset.validator import DatasetValidator, Severity


def test_valid_fixture_passes(validator: DatasetValidator, observations: pd.DataFrame) -> None:
    report = validator.validate(observations)

    assert report.passed, report.errors
    assert report.warnings == []
    assert report.row_count == len(observations)


def test_empty_dataset_fails(validator: DatasetValidator, observations: pd.DataFrame) -> None:
    report = validator.validate(observations.iloc[0:0])
    assert report.codes(Severity.ERROR) == {"empty_dataset"}


def test_missing_required_column(validator: DatasetValidator, observations: pd.DataFrame) -> None:
    report = validator.validate(observations.drop(columns=["SOIL_MOISTURE_10"]))

    assert not report.passed
    assert "missing_required_column" in report.codes(Severity.ERROR)


def test_missing_target_column(validator: DatasetValidator, observations: pd.DataFrame) -> None:
    report = validator.validate(observations.drop(columns=["SOIL_MOISTURE_5"]))
    assert "missing_target_column" in report.codes(Severity.ERROR)


def test_optional_column_may_be_absent(
    validator: DatasetValidator, observations: pd.DataFrame
) -> None:
    assert validator.validate(observations.drop(columns=["SOIL_MOISTURE_100"])).passed


def test_unexpected_column_is_a_warning(
    validator: DatasetValidator, observations: pd.DataFrame
) -> None:
    report = validator.validate(observations.assign(EXTRA=1.0))
    assert report.passed
    assert "unexpected_column" in report.codes(Severity.WARNING)


def test_incompatible_type(validator: DatasetValidator, observations: pd.DataFrame) -> None:
    frame = observations.assign(SOIL_MOISTURE_5=observations["SOIL_MOISTURE_5"].astype(str))
    assert "incompatible_type" in validator.validate(frame).codes(Severity.ERROR)


def test_duplicate_observation_keys(
    validator: DatasetValidator, observations: pd.DataFrame
) -> None:
    frame = pd.concat([observations, observations.iloc[[0, 1]]], ignore_index=True)

    report = validator.validate(frame)

    duplicate = next(issue for issue in report.errors if issue.code == "duplicate_observation_key")
    assert duplicate.count == 2


@pytest.mark.parametrize("bad_station", ["unknown station", "", "az_lower_1", "TX"])
def test_invalid_station_identifier(
    validator: DatasetValidator, observations: pd.DataFrame, bad_station: str
) -> None:
    frame = observations.copy()
    second_station = frame["station"].unique()[1]
    frame.loc[frame["station"] == second_station, "station"] = bad_station
    assert "invalid_station_id" in validator.validate(frame).codes(Severity.ERROR)


def test_station_inventory_is_enforced_by_profile(
    schema: DatasetSchema, observations: pd.DataFrame
) -> None:
    profile = ValidationProfile(
        name="inventory", expected_stations=(observations["station"].iloc[0], "NY_Gamma_3_E")
    )
    report = DatasetValidator(schema, profile).validate(observations)

    assert {"unknown_station", "missing_station"} <= report.codes(Severity.ERROR)


def test_inconsistent_station_coordinates(
    validator: DatasetValidator, observations: pd.DataFrame
) -> None:
    frame = observations.copy()
    frame.loc[0, "LATITUDE"] = 45.0
    assert "inconsistent_station_coordinates" in validator.validate(frame).codes(Severity.ERROR)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("SOIL_MOISTURE_5", 0.71),
        ("SOIL_MOISTURE_5", -0.01),
        ("RH_HR_AVG", 101.0),
        ("P_CALC", -1.0),
        ("LATITUDE", 91.0),
        ("LONGITUDE", -181.0),
    ],
)
def test_out_of_range_values(
    validator: DatasetValidator, observations: pd.DataFrame, column: str, value: float
) -> None:
    frame = observations.copy()
    frame[column] = frame[column].astype("float64")
    frame.loc[frame.index, column] = value
    report = validator.validate(frame)

    assert any(issue.code == "out_of_range" and issue.column == column for issue in report.errors)


def test_target_bounds_are_inclusive(
    validator: DatasetValidator, observations: pd.DataFrame
) -> None:
    frame = observations.copy()
    frame.loc[0, "SOIL_MOISTURE_5"] = 0.0
    frame.loc[1, "SOIL_MOISTURE_5"] = 0.70
    assert validator.validate(frame).passed


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_non_finite_values(
    validator: DatasetValidator, observations: pd.DataFrame, value: float
) -> None:
    frame = observations.copy()
    frame.loc[2, "SOIL_TEMP_5"] = value
    report = validator.validate(frame)

    assert any(
        issue.code == "non_finite_value" and issue.column == "SOIL_TEMP_5"
        for issue in report.errors
    )


def test_missing_target_observations_are_valid(
    validator: DatasetValidator, observations: pd.DataFrame
) -> None:
    frame = observations.copy()
    frame.loc[[0, 5, 7], "SOIL_MOISTURE_5"] = np.nan
    frame.loc[[1, 2], "P_CALC"] = np.nan

    report = validator.validate(frame)

    assert report.passed, report.errors
    info = next(issue for issue in report.issues if issue.code == "target_missing_values")
    assert info.severity is Severity.INFO
    assert info.count == 3


@pytest.mark.parametrize("column", ["time", "station", "LATITUDE", "annee"])
def test_null_in_mandatory_column(
    validator: DatasetValidator, observations: pd.DataFrame, column: str
) -> None:
    frame = observations.copy()
    if column == "annee":
        frame[column] = frame[column].astype("Int64")
    frame.loc[3, column] = None

    report = validator.validate(frame)

    assert any(
        issue.code == "null_in_mandatory_column" and issue.column == column
        for issue in report.errors
    )


def test_profile_dimensions(schema: DatasetSchema, observations: pd.DataFrame) -> None:
    profile = ValidationProfile(name="dims", expected_row_count=10, expected_column_count=5)
    report = DatasetValidator(schema, profile).validate(observations)
    assert {"row_count_mismatch", "column_count_mismatch"} <= report.codes(Severity.ERROR)


def test_validation_does_not_modify_the_dataframe(
    validator: DatasetValidator, observations: pd.DataFrame
) -> None:
    frame = observations.copy()
    frame.loc[0, "SOIL_MOISTURE_5"] = np.nan
    frame = frame.iloc[::-1]  # unordered input must stay unordered
    snapshot = frame.copy(deep=True)

    validator.validate(frame)

    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)
    assert frame["time"].dt.tz is None
