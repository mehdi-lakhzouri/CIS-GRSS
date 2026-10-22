from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable

import pandas as pd
import pytest

from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.scheduling.config import InsufficientPolicy, SchedulingConfig
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import (
    AttackWindowScheduler,
    ScheduleResult,
    SchedulingError,
)

ObservationFactory = Callable[..., pd.DataFrame]
SchedulingConfigFactory = Callable[..., SchedulingConfig]
SPIKE = AttackFamily.SPIKE
BIAS = AttackFamily.BIAS


def _plan(
    schema: DatasetSchema,
    partition_plan: PartitionPlan,
    config: SchedulingConfig,
    frame: pd.DataFrame,
) -> ScheduleResult:
    return AttackWindowScheduler(schema, partition_plan, config).plan(frame)


def _no_station_overlap(result: ScheduleResult) -> bool:
    by_station: dict[str, list[tuple]] = defaultdict(list)
    for window in result.windows:
        by_station[window.station].append((window.first_observation_time, window.end_exclusive))
    for intervals in by_station.values():
        intervals.sort()
        for (_, end), (start, _) in itertools.pairwise(intervals):
            if start < end:
                return False
    return True


def _window_signature(result: ScheduleResult) -> list[tuple]:
    return [
        (w.window_id, w.family.value, w.station, w.first_observation_time, w.row_labels)
        for w in result.windows
    ]


def test_selects_valid_windows(
    schema, partition_plan, observations, make_scheduling_config
) -> None:
    result = _plan(schema, partition_plan, make_scheduling_config(), observations)

    assert result.windows
    labels_in_frame = set(observations.index)
    for window in result.windows:
        assert window.partition == "validation"
        assert 2 <= window.length <= 6
        assert window.length == len(window.row_labels)
        assert set(window.row_labels) <= labels_in_frame
        assert window.end_exclusive > window.last_observation_time


def test_plan_is_deterministic(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=168)
    config = make_scheduling_config(attack_budget=0.2)
    first = _plan(schema, partition_plan, config, frame)
    second = _plan(schema, partition_plan, config, frame)
    assert _window_signature(first) == _window_signature(second)


def test_different_seed_changes_placement(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=168)
    a = _plan(
        schema, partition_plan, make_scheduling_config(global_seed=14, attack_budget=0.2), frame
    )
    b = _plan(
        schema, partition_plan, make_scheduling_config(global_seed=99, attack_budget=0.2), frame
    )
    assert _window_signature(a) != _window_signature(b)
    # Same budget denominator, so the same total is requested regardless of seed.
    assert a.requested_quota == b.requested_quota


def test_no_overlap_within_station(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=168)
    result = _plan(schema, partition_plan, make_scheduling_config(attack_budget=0.3), frame)
    assert _no_station_overlap(result)


def test_independent_stations_are_scheduled(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=168, stations=("AZ_Alpha_1_N", "OK_Beta_2_SE"))
    result = _plan(schema, partition_plan, make_scheduling_config(attack_budget=0.2), frame)
    assert {window.station for window in result.windows} == {"AZ_Alpha_1_N", "OK_Beta_2_SE"}


def test_windows_respect_partition_boundaries(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    # Observations straddling the 2024/2025 boundary; scope both partitions.
    frame = make_observations(start="2024-12-20 00:00", hours=600, stations=("AZ_Alpha_1_N",))
    config = make_scheduling_config(
        budget_scope=("validation", "test"), attack_budget=0.3, families=(SPIKE,)
    )
    result = _plan(schema, partition_plan, config, frame)

    assert result.windows
    assert {window.partition for window in result.windows} <= {"validation", "test"}
    for window in result.windows:
        # The measured interval of every observation must lie in the window's partition.
        interval_start_years = {
            (time - pd.Timedelta(hours=1)).year
            for time in (window.first_observation_time, window.last_observation_time)
        }
        expected_year = {2024} if window.partition == "validation" else {2025}
        assert interval_start_years <= expected_year


def test_leap_day_window_is_allowed(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(start="2024-02-28 00:00", hours=96, stations=("AZ_Alpha_1_N",))
    result = _plan(schema, partition_plan, make_scheduling_config(attack_budget=0.3), frame)
    assert result.windows
    assert all(window.partition == "validation" for window in result.windows)


def test_window_length_bounds_are_respected(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=168)
    config = make_scheduling_config(attack_budget=0.3, min_window_length=4, max_window_length=4)
    result = _plan(schema, partition_plan, config, frame)
    assert result.windows
    assert {window.length for window in result.windows} == {4}


def test_first_and_last_positions_are_usable(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    # Window length equal to the whole segment: the only placement covers it entirely.
    frame = make_observations(start="2024-03-01 01:00", hours=5, stations=("AZ_Alpha_1_N",))
    config = make_scheduling_config(
        attack_budget=1.0, families=(SPIKE,), min_window_length=5, max_window_length=5
    )
    result = _plan(schema, partition_plan, config, frame)

    assert len(result.windows) == 1
    window = result.windows[0]
    assert window.first_observation_time == pd.Timestamp("2024-03-01 01:00", tz="UTC")
    assert window.last_observation_time == pd.Timestamp("2024-03-01 05:00", tz="UTC")


def test_history_requirement_blocks_early_starts(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(start="2024-03-01 01:00", hours=5, stations=("AZ_Alpha_1_N",))
    config = make_scheduling_config(
        attack_budget=1.0,
        families=(SPIKE,),
        min_window_length=5,
        max_window_length=5,
        min_history={SPIKE: 1},
    )
    result = _plan(schema, partition_plan, config, frame)
    assert result.windows == ()  # no room for history + a length-5 window in 5 rows


def test_zero_budget_schedules_nothing(
    schema, partition_plan, observations, make_scheduling_config
) -> None:
    result = _plan(schema, partition_plan, make_scheduling_config(attack_budget=0.0), observations)
    assert result.requested_quota == 0
    assert result.windows == ()
    assert result.ledger.total_allocated == 0


def test_low_budget_stays_within_quota(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=168)
    result = _plan(schema, partition_plan, make_scheduling_config(attack_budget=0.02), frame)
    assert 0 < result.reserved_total <= result.requested_quota


def test_no_overshoot_per_cell_and_total(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=168)
    result = _plan(schema, partition_plan, make_scheduling_config(attack_budget=0.3), frame)
    assert result.reserved_total <= result.ledger.total_allocated <= result.requested_quota
    for cell_ledger in result.ledger.cell_ledgers():
        assert cell_ledger.reserved <= cell_ledger.allocated


def test_quota_distribution_follows_weights(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=336)
    config = make_scheduling_config(
        attack_budget=0.3, families=(SPIKE, BIAS), weights={SPIKE: 3.0, BIAS: 1.0}
    )
    result = _plan(schema, partition_plan, config, frame)
    assert sum(result.family_quotas.values()) == result.requested_quota
    assert result.family_quotas[SPIKE] > result.family_quotas[BIAS]


def test_insufficient_eligible_windows_reports_deficit(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=10, stations=("AZ_Alpha_1_N",))
    config = make_scheduling_config(
        attack_budget=1.0, families=(SPIKE,), min_window_length=20, max_window_length=20
    )
    result = _plan(schema, partition_plan, config, frame)
    assert result.windows == ()
    assert result.unfulfilled_total == result.ledger.total_allocated > 0
    assert result.rejections


def test_strict_policy_raises_on_deficit(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=10, stations=("AZ_Alpha_1_N",))
    config = make_scheduling_config(
        attack_budget=1.0,
        families=(SPIKE,),
        min_window_length=20,
        max_window_length=20,
        insufficient_policy=InsufficientPolicy.FAIL,
    )
    with pytest.raises(SchedulingError, match="could not be reserved"):
        _plan(schema, partition_plan, config, frame)


def test_reservations_can_be_cancelled_and_committed(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=168)
    result = _plan(schema, partition_plan, make_scheduling_config(attack_budget=0.2), frame)
    window = result.windows[0]
    ledger = result.ledger
    reserved_before = ledger.total_reserved

    # Phase 3 would commit the number actually modified; reserved != committed.
    assert ledger.total_committed == 0
    ledger.commit(window.window_id, window.length)
    assert ledger.total_committed == window.length

    other = result.windows[1]
    ledger.cancel(other.window_id)
    assert ledger.total_reserved == reserved_before - other.length


def test_reserved_is_distinct_from_modified(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=168)
    result = _plan(schema, partition_plan, make_scheduling_config(attack_budget=0.2), frame)
    assert result.reserved_total > 0
    assert result.ledger.total_committed == 0


def test_plan_does_not_mutate_the_frame(
    schema, partition_plan, make_observations, make_scheduling_config
) -> None:
    frame = make_observations(hours=168)
    snapshot = frame.copy(deep=True)
    _plan(schema, partition_plan, make_scheduling_config(attack_budget=0.3), frame)
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)
