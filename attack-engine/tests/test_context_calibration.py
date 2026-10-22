from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from attack_engine.attacks.calibration import (
    CalibrationError,
    fit_calibration,
    load_calibration,
    save_calibration,
    training_features,
)
from attack_engine.attacks.config import StealthParameters


def test_fit_is_train_only_even_when_future_targets_and_context_are_corrupted(
    stealth_frame, schema, partition_plan, stealth_parameters
):
    original = stealth_frame.copy(deep=True)
    first = fit_calibration(stealth_frame, schema, partition_plan, stealth_parameters)
    future = stealth_frame.copy(deep=True)
    future.loc[
        future.time.dt.year.ge(2024),
        [schema.target_column, *stealth_parameters.required_context_features],
    ] = -99999
    second = fit_calibration(future, schema, partition_plan, stealth_parameters)
    assert first == second
    assert first.training_sha256 == second.training_sha256
    assert first.calibration_sha256 == first.computed_sha256()
    assert all(model.last_training_time.year == 2023 for model in first.station_models.values())
    pd.testing.assert_frame_equal(stealth_frame, original, check_exact=True)


def test_calibration_is_deterministic_and_row_order_independent(
    stealth_frame, schema, partition_plan, stealth_parameters
):
    first = fit_calibration(stealth_frame, schema, partition_plan, stealth_parameters)
    second = fit_calibration(stealth_frame.iloc[::-1], schema, partition_plan, stealth_parameters)
    assert first == second


def test_ridge_recovers_an_explainable_contextual_relationship(
    make_observations, schema, partition_plan, stealth_parameters
):
    frame = make_observations(start="2023-06-01 01:00", hours=240, stations=("AZ_Alpha_1_N",))
    rng = np.random.default_rng(10)
    frame.SOIL_MOISTURE_10 = rng.uniform(0.1, 0.5, len(frame))
    frame.SOIL_MOISTURE_5 = 0.1 + 0.7 * frame.SOIL_MOISTURE_10
    params = stealth_parameters.model_copy(update={"ridge_alpha": 1e-6})
    artifact = fit_calibration(frame, schema, partition_plan, params)
    model = artifact.station_models["AZ_Alpha_1_N"]
    assert model.coefficients[0] / model.feature_scales[0] == pytest.approx(0.7, abs=1e-5)
    features = training_features(frame, schema, params).iloc[-1].to_numpy()
    assert model.predict(features) == pytest.approx(frame.SOIL_MOISTURE_5.iloc[-1], abs=1e-5)
    assert model.residual_tolerance >= params.residual_floor


def test_no_training_partition_rows_is_an_error(
    stealth_frame, schema, partition_plan, stealth_parameters
):
    with pytest.raises(CalibrationError, match="no observations"):
        fit_calibration(
            stealth_frame.loc[stealth_frame.time.dt.year.eq(2024)],
            schema,
            partition_plan,
            stealth_parameters,
        )


def test_missing_context_station_has_no_implicit_fallback(
    stealth_frame, schema, partition_plan, stealth_parameters
):
    frame = stealth_frame.copy(deep=True)
    frame.loc[frame.station.eq("OK_Beta_2_SE"), "P_CALC"] = np.nan
    artifact = fit_calibration(frame, schema, partition_plan, stealth_parameters)
    assert "OK_Beta_2_SE" not in artifact.station_models
    assert "OK_Beta_2_SE" in artifact.unavailable_stations
    frame.P_CALC = np.nan
    with pytest.raises(CalibrationError, match="no station"):
        fit_calibration(frame, schema, partition_plan, stealth_parameters)


def test_training_features_never_use_the_current_or_future_target(
    make_observations, schema, stealth_parameters
):
    frame = make_observations(hours=30, stations=("AZ_Alpha_1_N",))
    before = training_features(frame, schema, stealth_parameters)
    frame.loc[12:, schema.target_column] = 0.65
    after = training_features(frame, schema, stealth_parameters)
    pd.testing.assert_frame_equal(before.iloc[:13], after.iloc[:13], check_exact=True)
    assert not before.iloc[13:].equals(after.iloc[13:])


def test_missing_target_and_time_gap_reset_causal_history(
    make_observations, schema, stealth_parameters
):
    frame = make_observations(hours=30, stations=("AZ_Alpha_1_N",))
    frame.loc[10, schema.target_column] = np.nan
    features = training_features(frame, schema, stealth_parameters)
    assert features.loc[11:14, "target_history_mean"].isna().all()
    assert np.isfinite(features.loc[15, "target_history_mean"])
    frame = frame.drop(index=20)
    features = training_features(frame, schema, stealth_parameters)
    assert features.loc[21:24, "target_history_mean"].isna().all()


def test_year_end_observation_remains_authorized_train(
    make_observations, schema, partition_plan, stealth_parameters
):
    frame = make_observations(start="2023-12-31 20:00", hours=10, stations=("AZ_Alpha_1_N",))
    params = stealth_parameters.model_copy(update={"min_history_hours": 1, "min_training_rows": 2})
    artifact = fit_calibration(frame, schema, partition_plan, params)
    model = artifact.station_models["AZ_Alpha_1_N"]
    assert model.training_rows == 5
    assert model.last_training_time.isoformat() == "2024-01-01T00:00:00+00:00"


def test_artifact_roundtrip_and_digest(stealth_calibration, stealth_parameters):
    assert load_calibration(stealth_parameters.calibration_path) == stealth_calibration
    original = stealth_parameters.calibration_path.read_bytes()
    save_calibration(stealth_calibration, stealth_parameters.calibration_path)
    assert stealth_parameters.calibration_path.read_bytes() == original


def test_tampering_is_rejected(stealth_calibration, stealth_parameters):
    path = stealth_parameters.calibration_path
    document = json.loads(path.read_text())
    document["station_models"]["AZ_Alpha_1_N"]["intercept"] += 0.01
    path.write_text(json.dumps(document))
    with pytest.raises(CalibrationError, match="SHA-256 mismatch"):
        load_calibration(path)


def test_artifact_cannot_be_written_as_a_dataset(stealth_calibration, tmp_path):
    path = tmp_path / "original.parquet"
    path.write_bytes(b"original")
    with pytest.raises(CalibrationError, match="json"):
        save_calibration(stealth_calibration, path)
    assert path.read_bytes() == b"original"


def test_incompatible_artifact_is_rejected(
    stealth_calibration, stealth_parameters, schema, partition_plan
):
    different = stealth_parameters.model_copy(update={"min_history_hours": 12})
    with pytest.raises(CalibrationError, match="does not match"):
        stealth_calibration.require_compatible(different, schema, partition_plan)


@pytest.mark.parametrize(
    "changes",
    [
        {"magnitude_min": 0.1, "magnitude_max": 0.01},
        {"max_change_rate": 0},
        {"min_attack_impact": 0},
        {"min_history_hours": 0},
        {"max_cumulative_drift": -1},
        {"max_step_change": float("inf")},
        {"required_context_features": ["SOIL_MOISTURE_5"]},
        {"required_context_features": ["P_CALC", "P_CALC"]},
        {"required_context_features": []},
        {"attack_partitions": ["train"]},
        {"attack_partitions": []},
        {"max_candidate_attempts": 0},
        {"missing_context_policy": "fill_zero"},
        {"min_attack_impact": 0.5},
    ],
)
def test_stealth_parameter_validation(stealth_parameters, changes):
    data = stealth_parameters.model_dump()
    data.update(changes)
    with pytest.raises(ValidationError):
        StealthParameters.model_validate(data)
