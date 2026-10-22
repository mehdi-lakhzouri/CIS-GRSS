from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.scheduling.eligibility import EligibilityEngine, EligibilityError
from attack_engine.scheduling.models import RejectionReason, ScopeCell

ObservationFactory = Callable[..., pd.DataFrame]


def _engine(
    schema: DatasetSchema, partition_plan: PartitionPlan, scope: tuple[str, ...] = ("validation",)
) -> EligibilityEngine:
    return EligibilityEngine(schema, partition_plan, scope)


def test_full_fixture_is_one_segment_per_cell(
    schema: DatasetSchema, partition_plan: PartitionPlan, observations: pd.DataFrame
) -> None:
    report = _engine(schema, partition_plan).analyze(observations)

    assert len(report.cells) == 2
    for cell in report.cells:
        assert cell.finite_target_count == 48
        assert len(cell.segments) == 1
        assert cell.segments[0].length == 48
        assert cell.eligible_observation_count == 48
    assert report.rejections == ()


def test_missing_target_splits_segment_without_shrinking_denominator(
    schema: DatasetSchema, partition_plan: PartitionPlan, observations: pd.DataFrame
) -> None:
    frame = observations.copy()
    first_station = frame["station"].unique()[0]
    mask = frame["station"] == first_station
    gap_index = frame.index[mask][20]
    frame.loc[gap_index, "SOIL_MOISTURE_5"] = np.nan

    cell = (
        _engine(schema, partition_plan)
        .analyze(frame)
        .cell(ScopeCell(station=first_station, partition="validation"))
    )

    assert cell is not None
    assert cell.finite_target_count == 47  # denominator excludes the missing target
    assert cell.missing_target_count == 1
    assert len(cell.segments) == 2
    assert {segment.length for segment in cell.segments} == {20, 27}
    assert cell.eligible_observation_count == 47


def test_time_gap_breaks_continuity(
    schema: DatasetSchema, partition_plan: PartitionPlan, observations: pd.DataFrame
) -> None:
    frame = observations.copy()
    first_station = frame["station"].unique()[0]
    drop = frame.index[frame["station"] == first_station][10]
    frame = frame.drop(index=drop)

    cell = (
        _engine(schema, partition_plan)
        .analyze(frame)
        .cell(ScopeCell(station=first_station, partition="validation"))
    )

    assert cell is not None
    assert len(cell.segments) == 2
    assert cell.finite_target_count == 47
    assert cell.eligible_observation_count == 47


def test_scope_restricts_partitions(
    schema: DatasetSchema, partition_plan: PartitionPlan, make_observations: ObservationFactory
) -> None:
    # 4 hours across the 2024/2025 boundary: 3 in validation, 1 in test.
    frame = make_observations(start="2024-12-31 22:00", hours=4, stations=("AZ_Alpha_1_N",))

    validation_only = _engine(schema, partition_plan, ("validation",)).analyze(frame)
    both = _engine(schema, partition_plan, ("validation", "test")).analyze(frame)

    assert {cell.cell.partition for cell in validation_only.cells} == {"validation"}
    assert validation_only.cells[0].finite_target_count == 3
    assert {cell.cell.partition for cell in both.cells} == {"validation", "test"}
    assert sum(cell.finite_target_count for cell in both.cells) == 4


def test_cell_with_no_continuous_segment_is_rejected(
    schema: DatasetSchema, partition_plan: PartitionPlan, make_observations: ObservationFactory
) -> None:
    # Single isolated finite observation with missing neighbours around it.
    frame = make_observations(start="2024-03-01 01:00", hours=3, stations=("AZ_Alpha_1_N",))
    frame.loc[frame.index[[0, 2]], "SOIL_MOISTURE_5"] = np.nan

    report = _engine(schema, partition_plan).analyze(frame)

    cell = report.cells[0]
    assert cell.finite_target_count == 1
    assert cell.segments[0].length == 1  # length-1 segment is still continuous
    frame.loc[frame.index[1], "SOIL_MOISTURE_5"] = np.nan  # now nothing finite
    report = _engine(schema, partition_plan).analyze(frame)
    assert report.cells[0].segments == ()
    assert any(r.reason is RejectionReason.NO_ELIGIBLE_SEGMENT for r in report.rejections)


def test_missing_required_column_raises(
    schema: DatasetSchema, partition_plan: PartitionPlan, observations: pd.DataFrame
) -> None:
    with pytest.raises(EligibilityError, match="SOIL_MOISTURE_5"):
        _engine(schema, partition_plan).analyze(observations.drop(columns=["SOIL_MOISTURE_5"]))


def test_analyze_does_not_mutate_the_frame(
    schema: DatasetSchema, partition_plan: PartitionPlan, observations: pd.DataFrame
) -> None:
    snapshot = observations.copy(deep=True)
    _engine(schema, partition_plan).analyze(observations)
    pd.testing.assert_frame_equal(observations, snapshot, check_exact=True)
