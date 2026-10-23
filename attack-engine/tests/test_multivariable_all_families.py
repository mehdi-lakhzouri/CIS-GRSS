"""Coordinated FDI contracts across the five existing attack families."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from attack_engine.attacks.config import (
    AttackExecutionConfig,
    BiasParameters,
    CrossVariableConstraint,
    DriftParameters,
    ReplayParameters,
    SpikeParameters,
    StealthParameters,
    VariableConstraints,
)
from attack_engine.attacks.executor import AttackExecutor, ExecutionError
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackSettings,
    MultiStealthContext,
    ReplayHistory,
    TargetBounds,
)
from attack_engine.attacks.multistealth import SecondaryCalibration, StationCoupling
from attack_engine.attacks.registry import transform_for
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler

SM5, SM10, RAIN = "SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC"
TARGETS = (SM5, SM10, RAIN)
SELECTIONS = [c for n in range(1, 4) for c in combinations(TARGETS, n)] + [(RAIN, SM10, SM5)]
FAMILIES = tuple(AttackFamily)


def parameters(family, targets, stealth_parameters):
    common = {"target_columns": targets}
    magnitudes = {SM5: 0.02, SM10: 0.01, RAIN: 0.5}
    additive = {
        **common,
        "sign": "positive",
        "secondary_magnitudes": {c: magnitudes[c] for c in targets[1:]},
    }
    if family is AttackFamily.SPIKE:
        return SpikeParameters(
            **additive, magnitude=magnitudes[targets[0]], min_points=3, max_points=3
        )
    if family is AttackFamily.BIAS:
        return BiasParameters(**additive, magnitude=magnitudes[targets[0]])
    if family is AttackFamily.DRIFT:
        return DriftParameters(**additive, final_magnitude=magnitudes[targets[0]])
    if family is AttackFamily.REPLAY:
        return ReplayParameters(**common, min_age_hours=12)
    return StealthParameters(
        **{
            **stealth_parameters.model_dump(),
            **common,
            "secondary_columns": tuple(c for c in targets if c != SM5),
            "rainfall_trigger": 0.001,
            "rainfall_lag_hours": 0,
        }
    )


@pytest.fixture
def multivariable_request(make_stealth_request):
    def build(family, targets=TARGETS):
        original = make_stealth_request(length=12, offset=24)
        clean = {SM5: original.clean_values, SM10: (0.3,) * 12, RAIN: (0.0,) * 12}
        source = {SM5: (0.18,) * 12, SM10: (0.28,) * 12, RAIN: (0.5,) * 12}
        start = original.window.first_observation_time - timedelta(hours=24)
        history = ReplayHistory(
            run_times=(tuple(start + timedelta(hours=i) for i in range(12)),),
            run_values=(source[targets[0]],),
            run_labels=(tuple(range(1000, 1012)),),
            run_secondary={c: (source[c],) for c in targets[1:]},
        )
        secondary = tuple(c for c in targets if c != SM5)
        calibration = SecondaryCalibration(
            secondary_columns=secondary,
            partition_start="2022-01-01",
            partition_end="2024-01-01",
            stations={
                original.window.station: StationCoupling(
                    depth_slope=0.5,
                    depth_rows=500,
                    rainfall_ceiling_mm=5.0,
                    rainfall_rows=50,
                )
            },
        )
        return replace(
            original,
            window=original.window.model_copy(update={"family": family}),
            clean_values=clean[targets[0]],
            target_columns=targets,
            extra_clean={c: clean[c] for c in targets[1:]},
            reference_clean=clean,
            history=history,
            multistealth_context=MultiStealthContext(
                secondary_clean={c: clean[c] for c in secondary},
                calibration=calibration,
            ),
        )

    return build


def assert_exact_result(result):
    for column in result.columns:
        clean, final = np.asarray(column.clean_values), np.asarray(column.candidate_values)
        mask = np.isfinite(clean) & np.isfinite(final) & (final != clean)
        np.testing.assert_array_equal(mask, column.changed_mask)
        np.testing.assert_array_equal(np.isnan(final), np.isnan(clean))
        assert result.per_variable_modifications[column.column] == int(mask.sum())
        values = final[mask]
        assert (values >= 0).all()
        assert (values <= (200 if column.column == RAIN else 0.7)).all()
    assert result.modified_cells == sum(result.per_variable_modifications.values())
    assert result.modified_rows == len(result.modified_row_labels())
    assert len(result.modified_cell_keys()) == len(set(result.modified_cell_keys()))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("targets", SELECTIONS)
def test_every_family_accepts_individual_and_combined_targets(
    family,
    targets,
    multivariable_request,
    stealth_parameters,
):
    request = multivariable_request(family, targets)
    params = parameters(family, targets, stealth_parameters)
    transform = transform_for(family)
    result = transform(request, params, np.random.default_rng(14), AttackSettings())
    repeated = transform(request, params, np.random.default_rng(14), AttackSettings())
    assert result.applied, result.detail
    assert result == repeated
    assert result.changed_variables == targets
    assert set(result.per_variable_modifications) == set(targets)
    assert_exact_result(result)
    if family is AttackFamily.SPIKE:
        assert all(c.changed_mask == result.changed_mask for c in result.columns)
    if family is AttackFamily.BIAS:
        for c in result.columns:
            delta = np.asarray(c.candidate_values) - c.clean_values
            np.testing.assert_allclose(delta, delta[0])
    if family is AttackFamily.DRIFT:
        for c in result.columns:
            delta = np.asarray(c.candidate_values) - c.clean_values
            assert delta[0] == 0
            np.testing.assert_allclose(np.diff(delta), delta[-1] / 11)
    if family is AttackFamily.REPLAY:
        assert result.replay_source.last_time < request.window.first_observation_time


@pytest.mark.parametrize("family", FAMILIES)
def test_missing_secondary_values_are_preserved(family, multivariable_request, stealth_parameters):
    request = multivariable_request(family)
    extra = dict(request.extra_clean)
    extra[SM10] = tuple(np.nan if i == 5 else 0.3 for i in range(12))
    request = replace(
        request,
        extra_clean=extra,
        multistealth_context=replace(
            request.multistealth_context,
            secondary_clean={**request.multistealth_context.secondary_clean, SM10: extra[SM10]},
        ),
    )
    result = transform_for(family)(
        request,
        parameters(family, TARGETS, stealth_parameters),
        np.random.default_rng(14),
        AttackSettings(),
    )
    assert result.applied, result.detail
    assert_exact_result(result)
    depth = next(c for c in result.columns if c.column == SM10)
    assert np.isnan(depth.candidate_values[5]) and not depth.changed_mask[5]


@pytest.mark.parametrize("family", FAMILIES)
def test_infeasible_cross_relationship_rejects_entire_event(
    family,
    multivariable_request,
    stealth_parameters,
):
    request = multivariable_request(family)
    settings = AttackSettings(
        relationships=(
            CrossVariableConstraint(
                left=SM5,
                right=SM10,
                lower=0.1,
                upper=0.2,
            ),
        )
    )
    result = transform_for(family)(
        request,
        parameters(family, TARGETS, stealth_parameters),
        np.random.default_rng(14),
        settings,
    )
    assert result.rejection_reason is AttackRejectionReason.CROSS_VARIABLE_CONSTRAINT
    assert result.modified_cells == 0 and result.changed_variables == ()


@pytest.mark.parametrize("family", FAMILIES)
def test_temporal_limit_includes_window_exit(family, multivariable_request, stealth_parameters):
    request = replace(multivariable_request(family), next_values={RAIN: 0.0, SM10: 0.3, SM5: 0.2})
    settings = AttackSettings(
        variable_constraints={SM10: VariableConstraints(max_step_per_hour=1e-5)}
    )
    result = transform_for(family)(
        request,
        parameters(family, TARGETS, stealth_parameters),
        np.random.default_rng(14),
        settings,
    )
    assert result.rejection_reason is AttackRejectionReason.TEMPORAL_CONSTRAINT
    assert result.modified_cells == 0


def test_zero_secondary_effect_is_atomic(multivariable_request, stealth_parameters):
    request = multivariable_request(AttackFamily.STEALTH)
    calibration = request.multistealth_context.calibration
    station = request.window.station
    coupling = calibration.stations[station].model_copy(update={"depth_slope": 0.0})
    request = replace(
        request,
        multistealth_context=replace(
            request.multistealth_context,
            calibration=calibration.model_copy(update={"stations": {station: coupling}}),
        ),
    )
    result = transform_for(AttackFamily.STEALTH)(
        request,
        parameters(AttackFamily.STEALTH, TARGETS, stealth_parameters),
        np.random.default_rng(14),
        AttackSettings(),
    )
    assert result.rejection_reason is AttackRejectionReason.ZERO_EFFECT
    assert result.modified_cells == 0


@pytest.mark.parametrize("corruption", ["missing", "nan", "bounds", "future", "unaligned"])
def test_replay_requires_one_valid_past_block_for_all_variables(
    corruption,
    multivariable_request,
    stealth_parameters,
):
    request = multivariable_request(AttackFamily.REPLAY)
    history = request.history
    secondary = dict(history.run_secondary)
    if corruption == "missing":
        secondary.pop(RAIN)
    elif corruption in ("nan", "bounds"):
        secondary[RAIN] = ((np.nan if corruption == "nan" else 201.0,) * 12,)
    elif corruption == "future":
        history = replace(
            history, run_times=(tuple(t + timedelta(days=30) for t in history.run_times[0]),)
        )
    else:
        history = replace(
            history,
            run_times=(
                tuple(t + timedelta(minutes=i) for i, t in enumerate(history.run_times[0])),
            ),
        )
    request = replace(request, history=replace(history, run_secondary=secondary))
    result = transform_for(AttackFamily.REPLAY)(
        request,
        parameters(AttackFamily.REPLAY, TARGETS, stealth_parameters),
        np.random.default_rng(14),
        AttackSettings(),
    )
    assert result.rejection_reason is AttackRejectionReason.NO_REPLAY_SOURCE
    assert result.modified_cells == 0


def test_replay_copies_source_without_quantization(multivariable_request, stealth_parameters):
    request = multivariable_request(AttackFamily.REPLAY)
    rain = (0.123456789,) * 12
    request = replace(
        request,
        history=replace(
            request.history,
            run_secondary={**request.history.run_secondary, RAIN: (rain,)},
        ),
    )
    result = transform_for(AttackFamily.REPLAY)(
        request,
        parameters(AttackFamily.REPLAY, TARGETS, stealth_parameters),
        np.random.default_rng(14),
        AttackSettings(quantum=0.001),
    )
    assert result.applied
    assert next(c.candidate_values for c in result.columns if c.column == RAIN) == rain


def test_secondary_drift_has_its_own_slope_limit(multivariable_request, stealth_parameters):
    family = AttackFamily.DRIFT
    params = parameters(family, TARGETS, stealth_parameters).model_copy(
        update={"secondary_max_slopes_per_hour": {RAIN: 0.001}},
    )
    result = transform_for(family)(
        multivariable_request(family), params, np.random.default_rng(14), AttackSettings()
    )
    assert result.rejection_reason is AttackRejectionReason.SLOPE_TOO_STEEP


def test_unit_specific_precision_and_custom_bounds(multivariable_request, stealth_parameters):
    family = AttackFamily.BIAS
    settings = AttackSettings(
        bounds=TargetBounds(0.0, 0.25),
        quantum=0.001,
        variable_constraints={RAIN: VariableConstraints(quantum=0.1, high=1.0)},
    )
    request = multivariable_request(family)
    result = transform_for(family)(
        request,
        parameters(family, TARGETS, stealth_parameters),
        np.random.default_rng(14),
        settings,
    )
    assert result.applied
    infeasible = replace(settings, bounds=TargetBounds(0.0, 0.21))
    result = transform_for(family)(
        request,
        parameters(family, TARGETS, stealth_parameters),
        np.random.default_rng(14),
        infeasible,
    )
    assert result.rejection_reason is AttackRejectionReason.INFEASIBLE_BOUNDS


@pytest.mark.parametrize("unit", ["rows", "cells"])
@pytest.mark.parametrize("cap", ["max_modified_rows", "max_modified_cells"])
def test_both_budget_units_enforce_row_and_cell_caps(
    unit,
    cap,
    schema,
    partition_plan,
    make_observations,
    make_scheduling_config,
):
    family = AttackFamily.BIAS
    frame = make_observations(hours=96)
    snapshot = frame.copy(deep=True)
    scheduling = make_scheduling_config(
        families=(family,),
        attack_budget=0.8,
        min_window_length=4,
        max_window_length=4,
        variable_count={family: 3},
        budget_unit=unit,
        **{cap: 7 if cap.endswith("rows") else 15},
    )
    config = AttackExecutionConfig(
        global_seed=14,
        bias=BiasParameters(
            target_columns=TARGETS,
            magnitude=0.02,
            sign="positive",
            secondary_magnitudes={SM10: 0.01, RAIN: 0.5},
        ),
    )
    scheduler = AttackWindowScheduler(schema, partition_plan, scheduling)
    plan = scheduler.plan(frame)
    summary = AttackExecutor(schema, partition_plan, config).execute(frame, plan)
    assert summary.committed_rows == 4 and summary.committed_cells == 12
    assert summary.committed_total == (4 if unit == "rows" else 12)
    assert summary.committed_total <= summary.reserved_before <= summary.requested_quota
    assert summary.committed_total == sum(f.committed for f in summary.per_family.values())
    assert sum(summary.per_variable_modifications.values()) == 12
    assert summary.per_variable_modifications == dict.fromkeys(TARGETS, 4)
    assert any(r.rejection_reason is AttackRejectionReason.BUDGET_LIMIT for r in summary.results)
    assert all(c.active_reservation_count == 0 for c in plan.ledger.cell_ledgers())
    assert all(
        plan.ledger.reservation(r.window_id).committed_count
        == (r.modified_rows if unit == "rows" else r.modified_cells)
        for r in summary.applied_results
    )
    family_summary = summary.per_family[family]
    assert family_summary.unfulfilled_quota == family_summary.unreserved + family_summary.deficit
    assert family_summary.structural_deficit >= 0
    assert summary == AttackExecutor(schema, partition_plan, config).execute(
        frame, scheduler.plan(frame)
    )
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)


def test_bad_variable_reservation_fails_before_execution(
    schema,
    partition_plan,
    make_observations,
    make_scheduling_config,
):
    family = AttackFamily.SPIKE
    frame = make_observations(hours=96)
    scheduling = make_scheduling_config(families=(family,), attack_budget=0.5)
    config = AttackExecutionConfig(
        global_seed=14,
        spike=SpikeParameters(
            magnitude=0.1,
            target_columns=(SM5, RAIN),
            secondary_magnitudes={RAIN: 2.0},
        ),
    )
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    with pytest.raises(ExecutionError, match="scheduling reserved"):
        AttackExecutor(schema, partition_plan, config).execute(frame, plan)
    assert plan.ledger.total_committed == 0


def test_rainfall_only_eligibility_is_independent_of_missing_moisture(
    schema,
    partition_plan,
    make_observations,
    make_scheduling_config,
):
    family = AttackFamily.BIAS
    frame = make_observations(hours=96)
    frame[SM5] = np.nan
    scheduling = make_scheduling_config(
        families=(family,), attack_budget=0.5, eligibility_target=RAIN
    )
    config = AttackExecutionConfig(
        global_seed=14,
        bias=BiasParameters(
            magnitude=0.5,
            sign="positive",
            target_columns=(RAIN,),
        ),
    )
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    summary = AttackExecutor(schema, partition_plan, config).execute(frame, plan)
    assert summary.committed_total > 0
    assert set(summary.per_variable_modifications) == {RAIN}
    assert frame[SM5].isna().all()


@pytest.mark.parametrize(
    "targets,magnitudes",
    [
        ((), {}),
        ((SM5, SM5), {}),
        (("temperature",), {}),
        ((SM5, RAIN), {}),
        ((SM5,), {RAIN: 1}),
        ((SM5, RAIN), {RAIN: -1}),
        ((SM5, RAIN), {RAIN: np.inf}),
    ],
)
def test_invalid_target_or_unit_configuration_is_rejected(targets, magnitudes):
    with pytest.raises(ValidationError):
        SpikeParameters(magnitude=0.1, target_columns=targets, secondary_magnitudes=magnitudes)


def test_only_five_families_and_legacy_stealth_strategy_alias(stealth_parameters):
    assert {f.value for f in AttackFamily} == {"spike", "bias", "drift", "replay", "stealth"}
    data = stealth_parameters.model_dump(exclude={"target_columns", "secondary_columns"})
    params = StealthParameters(**data, secondary_columns=(SM10, RAIN))
    assert params.target_columns == TARGETS
    counts = Counter(params.target_columns)
    assert counts == dict.fromkeys(TARGETS, 1)


@pytest.mark.parametrize("family", FAMILIES)
def test_synthetic_executor_coordinates_every_family(
    family,
    stealth_frame,
    stealth_parameters,
    stealth_calibration,
    schema,
    partition_plan,
    make_scheduling_config,
):
    frame = stealth_frame.copy(deep=True)
    frame[SM10] = frame[SM5] * 0.8 + 0.05
    snapshot = frame.copy(deep=True)
    scheduling = make_scheduling_config(
        families=(family,),
        attack_budget=0.7,
        budget_scope=("validation", "test"),
        min_window_length=12,
        max_window_length=24,
        min_history={family: 12},
        variable_count={family: 3},
    )
    config = AttackExecutionConfig(
        global_seed=14,
        **{family.value: parameters(family, TARGETS, stealth_parameters)},
    )
    scheduler = AttackWindowScheduler(schema, partition_plan, scheduling)
    executor = AttackExecutor(schema, partition_plan, config)
    first = executor.execute(frame, scheduler.plan(frame))
    second = executor.execute(frame, scheduler.plan(frame))
    assert first == second
    assert first.applied_results
    assert first.committed_total == first.committed_cells > first.committed_rows
    assert sum(first.per_variable_modifications.values()) == first.committed_cells
    assert sum(f.committed for f in first.per_family.values()) == first.committed_total
    assert first.committed_total <= first.reserved_before <= first.requested_quota
    written = set()
    for result in first.results:
        assert result.partition != "train"
        assert_exact_result(result)
        assert written.isdisjoint(result.modified_cell_keys())
        written.update(result.modified_cell_keys())
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)


def test_mixed_target_counts_share_scheduler_and_budget_without_conflicts(
    stealth_frame,
    stealth_parameters,
    stealth_calibration,
    schema,
    partition_plan,
    make_scheduling_config,
):
    frame = stealth_frame.copy(deep=True)
    frame[SM10] = 0.8 * frame[SM5] + 0.05
    targets = dict(
        zip(FAMILIES, [(SM5, RAIN), (SM10,), TARGETS, TARGETS, (SM5, SM10)], strict=True)
    )
    scheduling = make_scheduling_config(
        families=FAMILIES,
        attack_budget=0.9,
        budget_scope=("validation", "test"),
        min_window_length=3,
        max_window_length=5,
        min_history=dict.fromkeys(FAMILIES, 12),
        variable_count={f: len(t) for f, t in targets.items()},
    )
    config = AttackExecutionConfig(
        global_seed=14, **{f.value: parameters(f, targets[f], stealth_parameters) for f in FAMILIES}
    )
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    summary = AttackExecutor(schema, partition_plan, config).execute(frame, plan)
    assert {r.family for r in summary.applied_results} == set(FAMILIES)
    assert sum(s.committed for s in summary.per_family.values()) == summary.committed_cells
    assert summary.committed_total <= summary.reserved_before <= summary.requested_quota
    keys = [k for r in summary.applied_results for k in r.modified_cell_keys()]
    assert len(keys) == len(set(keys)) == summary.committed_cells
    assert sum(summary.per_variable_modifications.values()) == summary.committed_cells


def test_bias_rejects_partial_quantized_offset(make_attack_request):
    request = make_attack_request(AttackFamily.BIAS, [0.201, 0.204])
    result = transform_for(AttackFamily.BIAS)(
        request,
        BiasParameters(magnitude=0.004, sign="positive"),
        np.random.default_rng(14),
        AttackSettings(quantum=0.01),
    )
    assert result.rejection_reason is AttackRejectionReason.INFEASIBLE_BOUNDS
    assert result.modified_cells == 0


def test_drift_checks_actual_quantized_slope(multivariable_request, stealth_parameters):
    family = AttackFamily.DRIFT
    params = parameters(family, TARGETS, stealth_parameters).model_copy(
        update={
            "secondary_max_slopes_per_hour": {SM10: 0.001},
        }
    )
    result = transform_for(family)(
        multivariable_request(family),
        params,
        np.random.default_rng(14),
        AttackSettings(variable_constraints={SM10: VariableConstraints(quantum=0.01)}),
    )
    assert result.rejection_reason is AttackRejectionReason.SLOPE_TOO_STEEP


def test_conflicting_write_check_is_atomic(multivariable_request, stealth_parameters):
    family = AttackFamily.BIAS
    result = transform_for(family)(
        multivariable_request(family),
        parameters(family, TARGETS, stealth_parameters),
        np.random.default_rng(14),
        AttackSettings(),
    )
    written = {result.modified_cell_keys()[-1]}
    before = set(written)
    with pytest.raises(ExecutionError, match="more than one window"):
        AttackExecutor._record_written(result, written)
    assert written == before


def test_secondary_lags_use_elapsed_hours_not_array_positions(stealth_parameters):
    from attack_engine.attacks.multistealth import _depth_candidate, _rainfall_candidate

    params = parameters(AttackFamily.STEALTH, TARGETS, stealth_parameters).model_copy(
        update={
            "depth_coupling_lag_hours": 2,
            "rainfall_lag_hours": 2,
        }
    )
    coupling = StationCoupling(
        depth_slope=0.5, depth_rows=100, rainfall_ceiling_mm=5, rainfall_rows=100
    )
    elapsed = np.array([0.0, 2.0, 4.0, 6.0])
    offsets = np.array([0.0, 0.02, 0.02, 0.02])
    depth = _depth_candidate(np.full(4, 0.3), offsets, coupling, params, elapsed)
    np.testing.assert_allclose(depth, [0.3, 0.3, 0.31, 0.31])
    rain = _rainfall_candidate(np.zeros(4), offsets, coupling, params, elapsed)
    np.testing.assert_allclose(rain, [1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("previous_offset,applied", [(0.01, True), (0.1, False)])
def test_temporal_continuity_uses_previous_event_offset(
    previous_offset,
    applied,
    multivariable_request,
    stealth_parameters,
):
    family = AttackFamily.BIAS
    request = replace(
        multivariable_request(family),
        previous_values={SM10: 0.3 + previous_offset},
        previous_offsets={SM10: previous_offset},
    )
    result = transform_for(family)(
        request,
        parameters(family, TARGETS, stealth_parameters),
        np.random.default_rng(14),
        AttackSettings(
            variable_constraints={SM10: VariableConstraints(max_offset_rate_per_hour=0.02)}
        ),
    )
    assert result.applied is applied
    if not applied:
        assert result.rejection_reason is AttackRejectionReason.TEMPORAL_CONSTRAINT
