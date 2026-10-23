from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from attack_engine.attacks import registry
from attack_engine.attacks.calibration import CalibrationError
from attack_engine.attacks.config import AttackConfigError
from attack_engine.attacks.executor import AttackExecutor
from attack_engine.attacks.models import AttackRejectionReason
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler

# The five families the baseline campaign executes.
BASELINE_FAMILIES = (
    AttackFamily.SPIKE,
    AttackFamily.BIAS,
    AttackFamily.DRIFT,
    AttackFamily.REPLAY,
    AttackFamily.STEALTH,
)


def test_all_five_families_are_causal_deterministic_and_reconcile(
    stealth_frame,
    stealth_parameters,
    stealth_calibration,
    schema,
    partition_plan,
    make_scheduling_config,
    make_attack_config,
    monkeypatch,
):
    frame = stealth_frame
    snapshot = frame.copy(deep=True)
    scheduling = make_scheduling_config(
        attack_budget=0.4,
        budget_scope=("validation", "test"),
        families=tuple(BASELINE_FAMILIES),
        min_window_length=6,
        max_window_length=12,
        min_history={AttackFamily.STEALTH: 4, AttackFamily.REPLAY: 12},
    )
    config = make_attack_config(
        quantum=0.001,
        spike={"magnitude": 0.01},
        bias={"magnitude": 0.01},
        drift={"final_magnitude": 0.01},
        replay={"min_age_hours": 1},
        stealth=stealth_parameters.model_dump(),
    )
    scheduler = AttackWindowScheduler(schema, partition_plan, scheduling)
    plan = scheduler.plan(frame)
    # Deliberately reverse planning order; the executor must still emit causally.
    plan = replace(plan, windows=tuple(reversed(plan.windows)))
    requests = []
    original = registry.ATTACK_TRANSFORMS[AttackFamily.STEALTH]

    def capture(request, params, rng, settings):
        requests.append(request)
        return original(request, params, rng, settings)

    monkeypatch.setitem(registry.ATTACK_TRANSFORMS, AttackFamily.STEALTH, capture)
    executor = AttackExecutor(schema, partition_plan, config)
    first = executor.execute(frame, plan)
    second_plan = scheduler.plan(frame)
    second_plan = replace(second_plan, windows=tuple(reversed(second_plan.windows)))
    second = executor.execute(frame, second_plan)
    assert first == second
    assert not first.skipped_families
    assert all(first.per_family[family].applied_windows > 0 for family in BASELINE_FAMILIES)
    modified = set()
    emissions = {}
    for result in first.results:
        clean = np.asarray(result.clean_values)
        values = np.asarray(result.candidate_values)
        np.testing.assert_array_equal(values != clean, result.changed_mask)
        assert result.modified_count == sum(result.changed_mask)
        for label, value, changed in zip(
            result.row_labels, values, result.changed_mask, strict=True
        ):
            if changed:
                assert label not in modified
                modified.add(label)
                emissions[label] = value
    observed_an_earlier_attack = False
    for request in requests:
        context = request.stealth_context
        earlier = frame.loc[
            frame.station.eq(request.window.station)
            & frame.time.lt(request.window.first_observation_time.replace(tzinfo=None))
        ]
        earlier = earlier.iloc[-stealth_calibration.history_samples :]
        expected = tuple(
            emissions.get(label, clean)
            for label, clean in zip(earlier.index, earlier.SOIL_MOISTURE_5, strict=True)
        )
        assert context.history_values == expected
        assert context.history_times[-1] < request.window.first_observation_time
        observed_an_earlier_attack |= expected != tuple(earlier.SOIL_MOISTURE_5)
    assert observed_an_earlier_attack
    assert len(modified) == first.committed_total == plan.ledger.total_committed
    assert plan.ledger.total_reserved == first.committed_total
    assert first.requested_quota - first.committed_total == first.deficit_vs_quota
    assert first.reserved_before - first.committed_total == first.released_total
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)


def test_stealth_is_not_implicitly_skipped_when_its_parameters_are_missing(
    stealth_frame,
    schema,
    partition_plan,
    make_scheduling_config,
    make_attack_config,
):
    scheduling = make_scheduling_config(attack_budget=0.2, families=(AttackFamily.STEALTH,))
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(stealth_frame)
    reserved = plan.reserved_total
    with pytest.raises(AttackConfigError, match="stealth"):
        AttackExecutor(schema, partition_plan, make_attack_config()).execute(stealth_frame, plan)
    assert plan.reserved_total == reserved
    assert plan.ledger.total_committed == 0


def test_missing_artifact_fails_before_changing_ledger(
    stealth_frame,
    stealth_parameters,
    schema,
    partition_plan,
    make_scheduling_config,
    make_attack_config,
):
    scheduling = make_scheduling_config(attack_budget=0.2, families=(AttackFamily.STEALTH,))
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(stealth_frame)
    config = make_attack_config(stealth=stealth_parameters.model_dump())
    reserved = plan.reserved_total
    with pytest.raises(CalibrationError, match="cannot load"):
        AttackExecutor(schema, partition_plan, config).execute(stealth_frame, plan)
    assert plan.reserved_total == reserved
    assert plan.ledger.total_committed == 0


def test_train_windows_are_rejected_and_released_without_reallocating_quota(
    stealth_frame,
    stealth_parameters,
    stealth_calibration,
    schema,
    partition_plan,
    make_scheduling_config,
    make_attack_config,
):
    scheduling = make_scheduling_config(
        attack_budget=0.1, families=(AttackFamily.STEALTH,), budget_scope=("train",)
    )
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(stealth_frame)
    quota = plan.requested_quota
    config = make_attack_config(stealth=stealth_parameters.model_dump())
    summary = AttackExecutor(schema, partition_plan, config).execute(stealth_frame, plan)
    assert summary.results
    assert all(
        r.rejection_reason is AttackRejectionReason.FORBIDDEN_PARTITION for r in summary.results
    )
    assert summary.committed_total == plan.ledger.total_reserved == 0
    assert summary.deficit_vs_quota == quota
    assert not summary.skipped_families
