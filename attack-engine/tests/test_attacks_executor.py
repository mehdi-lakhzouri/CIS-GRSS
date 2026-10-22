from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from attack_engine.attacks.config import AttackConfigError, AttackExecutionConfig
from attack_engine.attacks.executor import AttackExecutor, ExecutionSummary
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.scheduling.config import SchedulingConfig
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler, ScheduleResult

ObservationFactory = Callable[..., pd.DataFrame]
SchedulingConfigFactory = Callable[..., SchedulingConfig]
AttackConfigFactory = Callable[..., AttackExecutionConfig]

SPIKE = AttackFamily.SPIKE
BIAS = AttackFamily.BIAS
DRIFT = AttackFamily.DRIFT
REPLAY = AttackFamily.REPLAY


def _plan(
    schema: DatasetSchema,
    partition_plan: PartitionPlan,
    scheduling: SchedulingConfig,
    frame: pd.DataFrame,
) -> ScheduleResult:
    return AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)


def _execute(
    schema: DatasetSchema,
    partition_plan: PartitionPlan,
    attack_config: AttackExecutionConfig,
    frame: pd.DataFrame,
    plan: ScheduleResult,
) -> ExecutionSummary:
    return AttackExecutor(schema, partition_plan, attack_config).execute(frame, plan)


def _full_config(make_attack_config: AttackConfigFactory) -> AttackExecutionConfig:
    return make_attack_config(
        quantum=0.001,
        spike={"magnitude": 0.2, "min_points": 1, "max_points": 3},
        bias={"magnitude": 0.05},
        drift={"final_magnitude": 0.08, "max_slope_per_hour": 0.05},
        replay={"min_age_hours": 1},
    )


def test_execute_commits_actual_counts(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(
        attack_budget=0.3, families=(SPIKE, BIAS, DRIFT), min_history={}
    )
    plan = _plan(schema, partition_plan, scheduling, frame)
    summary = _execute(schema, partition_plan, _full_config(make_attack_config), frame, plan)

    assert summary.committed_total == plan.ledger.total_committed
    assert summary.committed_total <= summary.reserved_before <= summary.requested_quota
    assert summary.released_total == summary.reserved_before - summary.committed_total
    assert summary.frame_unchanged
    for execution in summary.per_family.values():
        assert execution.allocated >= execution.reserved >= execution.committed
        assert execution.unfulfilled_quota == execution.unreserved + execution.deficit
    assert sum(execution.unfulfilled_quota for execution in summary.per_family.values()) == (
        summary.deficit_vs_quota
    )


def test_spike_commits_fewer_than_reserved(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(attack_budget=0.3, families=(SPIKE,))
    plan = _plan(schema, partition_plan, scheduling, frame)
    summary = _execute(schema, partition_plan, _full_config(make_attack_config), frame, plan)

    spike = summary.per_family[SPIKE]
    # Spike changes 1-3 points per window but reserves the whole window length.
    assert spike.committed < spike.reserved
    assert spike.committed == sum(r.modified_count for r in summary.applied_results)


def test_clean_frame_is_unchanged(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    snapshot = frame.copy(deep=True)
    scheduling = make_scheduling_config(attack_budget=0.3, families=(SPIKE, BIAS, DRIFT))
    plan = _plan(schema, partition_plan, scheduling, frame)
    _execute(schema, partition_plan, _full_config(make_attack_config), frame, plan)
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)


def test_no_overlapping_writes_across_applied_windows(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(attack_budget=0.4, families=(SPIKE, BIAS, DRIFT))
    plan = _plan(schema, partition_plan, scheduling, frame)
    summary = _execute(schema, partition_plan, _full_config(make_attack_config), frame, plan)

    modified_labels = [
        label
        for result in summary.applied_results
        for label, changed in zip(result.row_labels, result.changed_mask, strict=True)
        if changed
    ]
    assert len(modified_labels) == len(set(modified_labels))


def test_no_cross_cell_modification(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336, stations=("AZ_Alpha_1_N", "OK_Beta_2_SE"))
    scheduling = make_scheduling_config(attack_budget=0.3, families=(BIAS,))
    plan = _plan(schema, partition_plan, scheduling, frame)
    summary = _execute(schema, partition_plan, _full_config(make_attack_config), frame, plan)

    label_to_cell = {label: (row.station, "validation") for label, row in frame.iterrows()}
    for result in summary.applied_results:
        for label in result.row_labels:
            assert label_to_cell[label][0] == result.station


def test_determinism(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(attack_budget=0.3, families=(SPIKE, BIAS, DRIFT))
    config = _full_config(make_attack_config)
    first = _execute(
        schema, partition_plan, config, frame, _plan(schema, partition_plan, scheduling, frame)
    )
    second = _execute(
        schema, partition_plan, config, frame, _plan(schema, partition_plan, scheduling, frame)
    )

    assert first.committed_total == second.committed_total
    assert [r.candidate_values for r in first.results] == [
        r.candidate_values for r in second.results
    ]


def test_future_unsupported_windows_are_skipped_and_released(
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
        families=(SPIKE, AttackFamily.STEALTH),
        weights={SPIKE: 1.0, AttackFamily.STEALTH: 1.0},
        min_history={AttackFamily.STEALTH: 0},
    )
    plan = _plan(schema, partition_plan, scheduling, frame)
    config = make_attack_config(spike={"magnitude": 0.2})
    summary = _execute(schema, partition_plan, config, frame, plan)

    assert AttackFamily.STEALTH in summary.skipped_families
    assert summary.per_family[AttackFamily.STEALTH].committed == 0
    # Stealth reservations were released, so committed never counts them.
    assert summary.committed_total == summary.per_family[SPIKE].committed


def test_missing_parameters_for_planned_family_fails(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=168)
    scheduling = make_scheduling_config(attack_budget=0.3, families=(SPIKE, BIAS))
    plan = _plan(schema, partition_plan, scheduling, frame)
    config = make_attack_config(spike={"magnitude": 0.2})  # bias params missing

    with pytest.raises(AttackConfigError, match="bias"):
        _execute(schema, partition_plan, config, frame, plan)


def test_replay_executes_with_sources(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=400)
    scheduling = make_scheduling_config(
        attack_budget=0.2,
        families=(REPLAY,),
        min_window_length=6,
        max_window_length=24,
        min_history={REPLAY: 48},
    )
    plan = _plan(schema, partition_plan, scheduling, frame)
    config = make_attack_config(replay={"min_age_hours": 1})
    summary = _execute(schema, partition_plan, config, frame, plan)

    windows = {window.window_id: window for window in plan.windows}
    replay = summary.per_family[REPLAY]
    assert replay.planned_windows > 0
    # Applied replay events carry a source strictly earlier than their window.
    for result in summary.applied_results:
        assert result.replay_source is not None
        assert result.replay_source.last_time < windows[result.window_id].first_observation_time
        assert result.replay_source.age_hours >= 1


def test_deficit_vs_quota_is_reported(
    schema, partition_plan, make_observations, make_scheduling_config, make_attack_config
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(attack_budget=0.3, families=(SPIKE,))
    plan = _plan(schema, partition_plan, scheduling, frame)
    summary = _execute(schema, partition_plan, _full_config(make_attack_config), frame, plan)
    assert summary.deficit_vs_quota == summary.requested_quota - summary.committed_total
    assert summary.deficit_vs_quota >= 0
