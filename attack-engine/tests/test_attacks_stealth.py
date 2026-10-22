from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from attack_engine.attacks.calibration import feature_vector
from attack_engine.attacks.config import MissingContextPolicy
from attack_engine.attacks.models import AttackRejectionReason, AttackSettings, TargetBounds
from attack_engine.attacks.stealth import apply_stealth

SETTINGS = AttackSettings(quantum=0.001)


def execute(request, params, seed=14, settings=SETTINGS):
    return apply_stealth(request, params, np.random.default_rng(seed), settings)


def test_stealth_obeys_all_constraints_and_uses_emitted_history(
    make_stealth_request, stealth_parameters
):
    request = make_stealth_request()
    result = execute(request, stealth_parameters)
    assert result.applied, result.detail
    clean = np.asarray(request.clean_values)
    final = np.asarray(result.candidate_values)
    np.testing.assert_array_equal(final != clean, result.changed_mask)
    assert result.modified_count == sum(result.changed_mask)
    assert (final >= 0).all() and (final <= 0.70).all()
    offsets = final - clean
    assert np.max(np.abs(np.diff(np.r_[0, offsets]))) <= stealth_parameters.max_change_rate + 1e-15
    assert np.abs(np.diff(np.r_[0, offsets])).sum() <= stealth_parameters.max_cumulative_drift
    assert np.mean(np.abs(offsets)) >= stealth_parameters.min_attack_impact
    context = request.stealth_context
    history = context.history_values
    model = context.calibration.station_models[request.window.station]
    for i, value in enumerate(final):
        features = feature_vector(
            tuple(
                context.measurements[name][i]
                for name in stealth_parameters.required_context_features
            ),
            history,
            context.times[i],
        )
        mu = model.predict(features)
        assert abs(value - mu) <= model.residual_tolerance + 1e-15
        history = (*history[1:], value)  # emitted candidate, NOT the clean target


@pytest.mark.parametrize("feature", ["P_CALC", "SOIL_MOISTURE_10", "SOIL_TEMP_5", "T_HR_AVG"])
def test_missing_context_is_not_filled_with_zero(make_stealth_request, stealth_parameters, feature):
    request = make_stealth_request()
    measurements = dict(request.stealth_context.measurements)
    measurements[feature] = (np.nan, *measurements[feature][1:])
    request = replace(
        request, stealth_context=replace(request.stealth_context, measurements=measurements)
    )
    result = execute(request, stealth_parameters)
    assert result.rejection_reason is AttackRejectionReason.MISSING_CONTEXT
    assert result.candidate_values == request.clean_values
    assert result.diagnostics.extra["context_available"] == request.length - 1


def test_skip_observation_is_explicit_and_retains_that_exact_value(
    make_stealth_request, stealth_parameters
):
    request = make_stealth_request()
    measurements = dict(request.stealth_context.measurements)
    measurements["P_CALC"] = (np.nan, *measurements["P_CALC"][1:])
    request = replace(
        request, stealth_context=replace(request.stealth_context, measurements=measurements)
    )
    params = stealth_parameters.model_copy(
        update={"missing_context_policy": MissingContextPolicy.SKIP_OBSERVATION}
    )
    result = execute(request, params)
    assert result.applied, result.detail
    assert result.candidate_values[0] == request.clean_values[0]
    assert result.changed_mask[0] is False


def test_insufficient_history_is_rejected(make_stealth_request, stealth_parameters):
    result = execute(make_stealth_request(offset=1), stealth_parameters)
    assert result.rejection_reason is AttackRejectionReason.INSUFFICIENT_HISTORY


def test_calibration_partition_is_never_attacked(make_stealth_request, stealth_parameters):
    result = execute(make_stealth_request(partition="train"), stealth_parameters)
    assert result.rejection_reason is AttackRejectionReason.FORBIDDEN_PARTITION


def test_missing_target_is_preserved(make_stealth_request, stealth_parameters):
    request = make_stealth_request()
    request = replace(request, clean_values=(np.nan, *request.clean_values[1:]))
    result = execute(request, stealth_parameters)
    assert result.rejection_reason is AttackRejectionReason.MISSING_TARGET
    assert np.isnan(result.candidate_values[0])
    assert not any(result.changed_mask)


def test_empty_feasible_range_is_explicit(make_stealth_request, stealth_parameters):
    request = make_stealth_request()
    artifact = request.stealth_context.calibration
    model = artifact.station_models[request.window.station].model_copy(
        update={
            "intercept": 0.9,
            "coefficients": (0.0,) * len(artifact.feature_names),
            "residual_tolerance": 0.001,
        }
    )
    artifact = artifact.model_copy(update={"station_models": {request.window.station: model}})
    request = replace(
        request, stealth_context=replace(request.stealth_context, calibration=artifact)
    )
    result = execute(request, stealth_parameters)
    assert result.rejection_reason is AttackRejectionReason.INFEASIBLE_CONTEXT
    assert result.modified_count == 0
    assert result.candidate_values == request.clean_values
    assert result.diagnostics.extra["attempts"] == stealth_parameters.max_candidate_attempts


def test_minimum_impact_is_checked_after_quantization(make_stealth_request, stealth_parameters):
    params = stealth_parameters.model_copy(update={"min_attack_impact": 0.0099})
    result = execute(make_stealth_request(length=2), params)
    assert result.rejection_reason is AttackRejectionReason.BELOW_MIN_IMPACT


def test_quantization_cannot_escape_a_tight_rate_band(make_stealth_request, stealth_parameters):
    params = stealth_parameters.model_copy(update={"max_change_rate": 1e-5})
    result = execute(make_stealth_request(), params)
    assert not result.applied
    assert result.rejection_reason in {
        AttackRejectionReason.BELOW_MIN_IMPACT,
        AttackRejectionReason.QUANTIZATION_INFEASIBLE,
    }
    assert result.candidate_values == result.clean_values


def test_no_context_or_model_has_an_explicit_reason(make_stealth_request, stealth_parameters):
    request = make_stealth_request()
    result = execute(replace(request, stealth_context=None), stealth_parameters)
    assert result.rejection_reason is AttackRejectionReason.NO_CALIBRATION
    window = request.window.model_copy(update={"station": "AZ_Unknown_1_N"})
    result = execute(replace(request, window=window), stealth_parameters)
    assert result.rejection_reason is AttackRejectionReason.NO_CALIBRATION


def test_same_seed_reproduces_the_full_result(make_stealth_request, stealth_parameters):
    request = make_stealth_request()
    assert execute(request, stealth_parameters) == execute(request, stealth_parameters)


def test_future_clean_values_do_not_enter_earlier_predictions(
    make_stealth_request, stealth_parameters
):
    request = make_stealth_request()
    first = execute(request, stealth_parameters)
    changed = replace(
        request, clean_values=(*request.clean_values[:-1], request.clean_values[-1] + 0.001)
    )
    second = execute(changed, stealth_parameters)
    assert first.applied and second.applied
    assert first.candidate_values[:-1] == second.candidate_values[:-1]


def test_stealth_uses_actual_elapsed_hours(make_stealth_request, stealth_parameters):
    request = make_stealth_request(length=4)
    context = request.stealth_context
    elapsed = (0.0, 2.0, 5.0, 9.0)
    times = tuple(context.times[0] + timedelta(hours=hours) for hours in elapsed)
    window = request.window.model_copy(
        update={"last_observation_time": times[-1], "end_exclusive": times[-1] + timedelta(hours=1)}
    )
    request = replace(
        request, window=window, elapsed_hours=elapsed, stealth_context=replace(context, times=times)
    )
    result = execute(request, stealth_parameters)
    assert result.applied, result.detail
    offsets = np.asarray(result.candidate_values) - request.clean_values
    assert offsets[1] - offsets[0] > stealth_parameters.max_change_rate
    assert (offsets[1] - offsets[0]) / 2 <= stealth_parameters.max_change_rate + 1e-15


def test_final_values_respect_physical_bounds(make_stealth_request, stealth_parameters):
    request = make_stealth_request()
    result = execute(
        request,
        stealth_parameters,
        settings=AttackSettings(bounds=TargetBounds(0.0, 0.212), quantum=0.001),
    )
    assert result.applied, result.detail
    assert max(result.candidate_values) <= 0.212


def test_no_grid_point_in_a_feasible_interval_is_rejected():
    from attack_engine.attacks.stealth import _project

    assert _project(0.141, 0.1412, 0.1411, 0.1414, SETTINGS) is None
