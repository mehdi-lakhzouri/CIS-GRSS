"""Regression tests for the Phase 3 correctness audit.

Covers the quantization false-positive defect (a value already on the 0.001 grid,
re-quantized, shifts in float64 and was wrongly flagged as modified), the drift
held-first invariant, spike achievability accounting, and the explicit
unsupported-family execution policy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from attack_engine.attacks import base
from attack_engine.attacks.bias import apply_bias
from attack_engine.attacks.config import (
    BiasParameters,
    DriftParameters,
    ReplayParameters,
    SpikeParameters,
)
from attack_engine.attacks.drift import apply_drift
from attack_engine.attacks.executor import (
    AttackExecutor,
    ExecutionError,
    UnsupportedFamilyPolicy,
)
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackSettings,
    ReplayHistory,
    SignPolicy,
    TargetBounds,
)
from attack_engine.attacks.replay import apply_replay
from attack_engine.attacks.spike import apply_spike
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler

RequestFactory = Callable[..., AttackRequest]
Q_SETTINGS = AttackSettings(bounds=TargetBounds(0.0, 0.70), quantum=0.001)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# 1/2. Quantization must not manufacture changes ------------------------------------------


def test_requantizing_an_on_grid_value_is_not_a_change() -> None:
    # quantize(0.141) != 0.141 in float64; the changed mask must still say "unchanged".
    clean = np.array([0.141, 0.200, 0.303])
    candidate = clean.copy()  # no perturbation at all
    final, changed, saturated = base.finalize_candidate(clean, candidate, Q_SETTINGS)
    assert not changed.any()
    assert not saturated.any()
    np.testing.assert_array_equal(final, clean)


def test_legitimate_one_quantum_change_is_not_hidden() -> None:
    clean = np.array([0.141, 0.141])
    candidate = np.array([0.141, 0.142])  # exactly one quantum on the second point
    _, changed, _ = base.finalize_candidate(clean, candidate, Q_SETTINGS)
    assert changed.tolist() == [False, True]


@pytest.mark.parametrize("quantum", [None, 0.001, 0.003])
def test_zero_intent_preserves_even_off_grid_values(quantum: float | None) -> None:
    clean = np.array([0.141, 0.1414, 0.2496, 0.70])
    final, changed, saturated = base.finalize_candidate(
        clean, clean.copy(), AttackSettings(quantum=quantum)
    )
    np.testing.assert_array_equal(final, clean)
    assert not changed.any()
    assert not saturated.any()


@pytest.mark.parametrize("delta", [1e-12, np.spacing(0.141)])
def test_representable_small_attack_is_counted_without_quantization(delta: float) -> None:
    clean = np.array([0.141])
    candidate = clean + delta
    final, changed, _ = base.finalize_candidate(clean, candidate, AttackSettings())
    assert candidate[0] != clean[0]
    np.testing.assert_array_equal(final, candidate)
    assert changed.tolist() == [True]


def test_small_quantum_does_not_suppress_a_legitimate_small_attack() -> None:
    clean = np.array([0.141])
    final, changed, _ = base.finalize_candidate(clean, clean + 1e-10, AttackSettings(quantum=1e-12))
    assert changed.tolist() == [True]
    assert final[0] > clean[0]


def test_same_quantization_bin_preserves_clean_exactly() -> None:
    clean = np.array([0.141, 0.1414])
    final, changed, _ = base.finalize_candidate(clean, clean + 0.00001, Q_SETTINGS)
    np.testing.assert_array_equal(final, clean)
    assert not changed.any()


def test_sub_quantum_attack_crossing_a_bin_is_counted() -> None:
    clean = np.array([0.14149])
    final, changed, _ = base.finalize_candidate(clean, clean + 0.00002, Q_SETTINGS)
    assert changed.tolist() == [True]
    assert final[0] == pytest.approx(0.142)


def test_quantization_clipping_and_mask_describe_the_same_final_values() -> None:
    clean = np.array([0.141, 0.699, 0.70, 0.0])
    candidate = np.array([0.141, 0.9, 0.9, -0.1])
    final, changed, saturated = base.finalize_candidate(clean, candidate, Q_SETTINGS)
    np.testing.assert_array_equal(final, [0.141, 0.70, 0.70, 0.0])
    np.testing.assert_array_equal(changed, final != clean)
    assert changed.tolist() == saturated.tolist() == [False, True, False, False]


@pytest.mark.parametrize("quantum", [0.0, -0.001, float("nan"), float("inf")])
def test_quantization_rejects_an_invalid_quantum(quantum: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        base.quantize(np.array([0.141]), quantum)


def test_bias_small_offset_is_not_hidden(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.BIAS, [0.141, 0.25])
    result = apply_bias(
        request,
        BiasParameters(magnitude=1e-12, sign=SignPolicy.POSITIVE),
        _rng(),
        AttackSettings(),
    )
    assert result.applied
    assert result.changed_mask == (True, True)


def test_bias_does_not_accept_clipping_hidden_by_a_fixed_tolerance(
    make_attack_request: RequestFactory,
) -> None:
    request = make_attack_request(AttackFamily.BIAS, [0.141, 0.70 - 1e-10])
    result = apply_bias(
        request,
        BiasParameters(magnitude=2e-10, sign=SignPolicy.POSITIVE),
        _rng(),
        AttackSettings(),
    )
    assert result.rejection_reason is AttackRejectionReason.INFEASIBLE_BOUNDS


def test_bias_rejects_quantization_that_changes_only_some_positions(
    make_attack_request: RequestFactory,
) -> None:
    request = make_attack_request(AttackFamily.BIAS, [0.141, 0.14149])
    result = apply_bias(
        request,
        BiasParameters(magnitude=0.00002, sign=SignPolicy.POSITIVE),
        _rng(),
        Q_SETTINGS,
    )
    assert result.rejection_reason is AttackRejectionReason.INFEASIBLE_BOUNDS


def test_bias_sub_quantum_zero_effect_preserves_clean(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.BIAS, [0.141, 0.222])
    result = apply_bias(
        request,
        BiasParameters(magnitude=0.00001, sign=SignPolicy.POSITIVE),
        _rng(),
        Q_SETTINGS,
    )
    assert result.rejection_reason is AttackRejectionReason.ZERO_EFFECT
    assert result.candidate_values == request.clean_values


def test_spike_untouched_positions_not_flagged_under_quantization(
    make_attack_request: RequestFactory,
) -> None:
    # Clean values off the float grid; spike touches few positions, the rest must stay unchanged.
    request = make_attack_request(AttackFamily.SPIKE, [0.141, 0.222, 0.303, 0.144, 0.255, 0.166])
    params = SpikeParameters(magnitude=0.3, min_points=1, max_points=2)
    result = apply_spike(request, params, _rng(1), Q_SETTINGS)
    assert result.modified_count == sum(result.changed_mask)
    assert result.modified_count <= 2  # never the whole 6-row window


def test_replay_equal_values_not_flagged_under_quantization(
    make_attack_window, make_attack_request: RequestFactory
) -> None:
    window = make_attack_window(AttackFamily.REPLAY, start="2024-03-10 00:00", length=3)
    times = tuple(
        ts.to_pydatetime()
        for ts in pd.date_range("2024-03-01 00:00", periods=3, freq="h", tz="UTC")
    )
    # Source identical to destination (both 0.141 etc.) -> zero effect, not false positives.
    history = ReplayHistory(
        run_times=(times,), run_values=((0.141, 0.222, 0.303),), run_labels=((0, 1, 2),)
    )
    request = make_attack_request(
        AttackFamily.REPLAY, [0.141, 0.222, 0.303], window=window, history=history
    )
    result = apply_replay(request, ReplayParameters(min_age_hours=1), _rng(), Q_SETTINGS)
    assert not result.applied  # zero effect, not spurious changes


# 1. Drift held-first invariant -----------------------------------------------------------


def test_drift_holds_first_point_under_quantization(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.DRIFT, [0.141, 0.141, 0.141, 0.141, 0.141])
    params = DriftParameters(final_magnitude=0.1, sign=SignPolicy.POSITIVE, hold_first=True)
    result = apply_drift(request, params, _rng(), Q_SETTINGS)

    assert result.changed_mask[0] is False
    assert result.candidate_values[0] == result.clean_values[0] == 0.141
    assert result.modified_count <= request.window.length - 1


# 1. Drift invariant end to end (committed <= reserved - applied_windows) -----------------


def _drift_plan_and_execute(
    schema, partition_plan, make_scheduling_config, make_attack_config, frame
):
    scheduling = make_scheduling_config(
        attack_budget=0.4, families=(AttackFamily.DRIFT,), min_window_length=6, max_window_length=24
    )
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    config = make_attack_config(
        quantum=0.001, drift={"final_magnitude": 0.08, "max_slope_per_hour": 0.05}
    )
    return AttackExecutor(schema, partition_plan, config).execute(frame, plan)


def test_drift_execution_respects_held_first_invariant(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    summary = _drift_plan_and_execute(
        schema, partition_plan, make_scheduling_config, make_attack_config, frame
    )
    drift = summary.per_family[AttackFamily.DRIFT]
    assert drift.applied_windows > 0
    for result in summary.applied_results:
        assert result.changed_mask[0] is False
        assert result.candidate_values[0] == result.clean_values[0]
    # Each applied window leaves its first point unchanged.
    assert drift.committed <= drift.reserved - drift.applied_windows
    assert drift.max_modifiable == drift.reserved - drift.applied_windows


# 3. Spike achievability ------------------------------------------------------------------


def test_spike_committed_never_exceeds_max_modifiable(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(
        attack_budget=0.5, families=(AttackFamily.SPIKE,), min_window_length=1, max_window_length=6
    )
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    config = make_attack_config(spike={"magnitude": 0.2, "min_points": 1, "max_points": 3})
    summary = AttackExecutor(schema, partition_plan, config).execute(frame, plan)

    spike = summary.per_family[AttackFamily.SPIKE]
    # Spike can modify at most max_points per window, so reserving longer windows is unreachable.
    assert spike.committed <= spike.max_modifiable <= spike.reserved
    assert spike.structural_deficit == spike.reserved - spike.max_modifiable


def test_single_position_spike_windows_can_fulfil_the_modification_quota(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(
        attack_budget=0.3, families=(AttackFamily.SPIKE,), min_window_length=1, max_window_length=1
    )
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    config = make_attack_config(spike={"magnitude": 0.2, "min_points": 1, "max_points": 3})
    summary = AttackExecutor(schema, partition_plan, config).execute(frame, plan)
    spike = summary.per_family[AttackFamily.SPIKE]
    assert summary.committed_total == spike.max_modifiable == summary.requested_quota


def test_drift_capacity_excludes_its_zero_time_origin_even_with_hold_first_false(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(attack_budget=0.3, families=(AttackFamily.DRIFT,))
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    config = make_attack_config(drift={"final_magnitude": 0.08, "hold_first": False})
    summary = AttackExecutor(schema, partition_plan, config).execute(frame, plan)
    drift = summary.per_family[AttackFamily.DRIFT]
    assert drift.max_modifiable == drift.reserved - drift.planned_windows
    assert all(not result.changed_mask[0] for result in summary.applied_results)


# 4. Unsupported-family execution policy --------------------------------------------------


def test_unsupported_family_skip_is_explicit(
    schema,
    partition_plan,
    make_observations,
    make_scheduling_config,
    make_attack_config,
    monkeypatch,
) -> None:
    from attack_engine.attacks.config import EXECUTABLE_FAMILIES

    monkeypatch.setattr(
        "attack_engine.attacks.executor.EXECUTABLE_FAMILIES",
        tuple(f for f in EXECUTABLE_FAMILIES if f is not AttackFamily.STEALTH),
    )
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(
        attack_budget=0.3,
        families=(AttackFamily.SPIKE, AttackFamily.STEALTH),
        weights={AttackFamily.SPIKE: 1.0, AttackFamily.STEALTH: 1.0},
    )
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    config = make_attack_config(spike={"magnitude": 0.2})
    summary = AttackExecutor(
        schema, partition_plan, config, unsupported_policy=UnsupportedFamilyPolicy.SKIP
    ).execute(frame, plan)

    assert AttackFamily.STEALTH in summary.skipped_families
    assert summary.skipped_window_count == len(plan.windows_for(AttackFamily.STEALTH))
    assert summary.skipped_window_count > 0


def test_unsupported_family_error_policy_raises(
    schema,
    partition_plan,
    make_observations,
    make_scheduling_config,
    make_attack_config,
    monkeypatch,
) -> None:
    from attack_engine.attacks.config import EXECUTABLE_FAMILIES

    monkeypatch.setattr(
        "attack_engine.attacks.executor.EXECUTABLE_FAMILIES",
        tuple(f for f in EXECUTABLE_FAMILIES if f is not AttackFamily.STEALTH),
    )
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(
        attack_budget=0.3,
        families=(AttackFamily.SPIKE, AttackFamily.STEALTH),
        weights={AttackFamily.SPIKE: 1.0, AttackFamily.STEALTH: 1.0},
    )
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    config = make_attack_config(spike={"magnitude": 0.2})
    reserved_before = plan.ledger.total_reserved
    with pytest.raises(ExecutionError, match="non-executable"):
        AttackExecutor(
            schema, partition_plan, config, unsupported_policy=UnsupportedFamilyPolicy.ERROR
        ).execute(frame, plan)
    assert plan.ledger.total_reserved == reserved_before
    assert plan.ledger.total_committed == 0


def test_frame_unchanged_is_verified_by_checksum(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(attack_budget=0.3, families=(AttackFamily.SPIKE,))
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    config = make_attack_config(spike={"magnitude": 0.2})
    summary = AttackExecutor(schema, partition_plan, config).execute(frame, plan)
    assert summary.frame_unchanged is True
