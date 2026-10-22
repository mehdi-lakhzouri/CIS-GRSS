from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attack_engine.attacks.executor import ExecutionSummary
from attack_engine.attacks.models import AttackDiagnostics, AttackResult, AttackStatus
from attack_engine.exporting.builder import (
    assign_row_partitions,
    build_attacked_target,
    build_audit_dataset,
    build_detector_input,
    build_ground_truth,
)
from attack_engine.exporting.models import ExportError
from attack_engine.scheduling.models import AttackFamily


def _result(
    window_id: str,
    *,
    family: AttackFamily = AttackFamily.BIAS,
    station: str = "AZ_Alpha_1_N",
    partition: str = "validation",
    row_labels: tuple[int, ...],
    candidate_values: tuple[float, ...],
    changed_mask: tuple[bool, ...],
    status: AttackStatus = AttackStatus.APPLIED,
    rejection_reason=None,
) -> AttackResult:
    diagnostics = AttackDiagnostics(
        requested_magnitude=0.05,
        sign=1,
        modified_count=sum(changed_mask),
        mean_abs_delta=0.01,
        max_abs_delta=0.02,
        saturated_count=0,
    )
    return AttackResult(
        window_id=window_id,
        family=family,
        station=station,
        partition=partition,
        status=status,
        row_labels=row_labels,
        clean_values=tuple(0.2 for _ in row_labels),
        candidate_values=candidate_values,
        changed_mask=changed_mask,
        diagnostics=diagnostics,
        rejection_reason=rejection_reason,
    )


def _summary(results: tuple[AttackResult, ...]) -> ExecutionSummary:
    committed = sum(r.modified_count for r in results if r.applied)
    return ExecutionSummary(
        seed=0,
        requested_quota=0,
        reserved_before=0,
        committed_total=committed,
        released_total=0,
        per_family={},
        results=results,
        frame_unchanged=True,
    )


@pytest.fixture
def small_frame(make_observations):
    frame = make_observations(hours=12)
    # Make row 2 a missing target, like a real delivery gap.
    frame.loc[2, "SOIL_MOISTURE_5"] = np.nan
    return frame


def test_build_attacked_target_applies_changed_mask_only(small_frame) -> None:
    result = _result(
        "bias-x-0000",
        row_labels=(0, 1, 3),
        candidate_values=(0.9, 0.91, 0.92),
        changed_mask=(True, False, True),
    )
    summary = _summary((result,))

    from attack_engine.dataset.schema import uscrn_hourly_schema

    schema = uscrn_hourly_schema()
    attacked = build_attacked_target(small_frame, schema, summary)

    assert attacked.final_target.loc[0] == pytest.approx(0.9)
    assert attacked.is_attacked.loc[0]
    assert attacked.attack_id.loc[0] == "bias-x-0000"
    # Row 1 was reserved but not actually changed (changed_mask False): untouched.
    assert attacked.final_target.loc[1] == small_frame.loc[1, "SOIL_MOISTURE_5"]
    assert not attacked.is_attacked.loc[1]
    assert pd.isna(attacked.attack_id.loc[1])
    assert attacked.final_target.loc[3] == pytest.approx(0.92)


def test_zero_effect_attack_is_not_labeled(small_frame) -> None:
    from attack_engine.dataset.schema import uscrn_hourly_schema

    schema = uscrn_hourly_schema()
    result = _result(
        "bias-x-0001", row_labels=(4, 5), candidate_values=(0.2, 0.2), changed_mask=(False, False)
    )
    summary = _summary((result,))

    attacked = build_attacked_target(small_frame, schema, summary)

    assert not attacked.is_attacked.any()
    assert attacked.attack_id.isna().all()


def test_rejected_result_never_modifies_rows(small_frame) -> None:
    from attack_engine.dataset.schema import uscrn_hourly_schema

    schema = uscrn_hourly_schema()
    rejected = _result(
        "bias-x-0002",
        row_labels=(6, 7),
        candidate_values=(0.5, 0.5),
        changed_mask=(True, True),
        status=AttackStatus.REJECTED,
    )
    summary = _summary((rejected,))

    attacked = build_attacked_target(small_frame, schema, summary)

    assert not attacked.is_attacked.any()


def test_overlapping_committed_rows_raise(small_frame) -> None:
    from attack_engine.dataset.schema import uscrn_hourly_schema

    schema = uscrn_hourly_schema()
    first = _result("bias-x-0003", row_labels=(8,), candidate_values=(0.5,), changed_mask=(True,))
    second = _result(
        "drift-x-0004",
        family=AttackFamily.DRIFT,
        row_labels=(8,),
        candidate_values=(0.6,),
        changed_mask=(True,),
    )
    summary = _summary((first, second))

    with pytest.raises(ExportError, match="more than one"):
        build_attacked_target(small_frame, schema, summary)


def test_missing_target_preserved_in_ground_truth(small_frame, partition_plan) -> None:
    from attack_engine.dataset.schema import uscrn_hourly_schema

    schema = uscrn_hourly_schema()
    summary = _summary(())
    attacked = build_attacked_target(small_frame, schema, summary)
    partitions = assign_row_partitions(small_frame, schema, partition_plan)

    ground_truth = build_ground_truth(small_frame, schema, attacked, partitions)

    missing_row = ground_truth.iloc[2]
    assert pd.isna(missing_row["clean_target"])
    assert pd.isna(missing_row["attacked_target"])
    assert not missing_row["is_attacked"]
    assert not missing_row["is_evaluable"]


def test_build_audit_dataset_preserves_source_columns(small_frame, partition_plan) -> None:
    from attack_engine.dataset.schema import uscrn_hourly_schema

    schema = uscrn_hourly_schema()
    result = _result("bias-x-0005", row_labels=(0,), candidate_values=(0.99,), changed_mask=(True,))
    summary = _summary((result,))
    attacked = build_attacked_target(small_frame, schema, summary)
    partitions = assign_row_partitions(small_frame, schema, partition_plan)

    audit = build_audit_dataset(small_frame, schema, partition_plan, attacked, partitions)

    pd.testing.assert_series_equal(
        audit["SOIL_MOISTURE_5"].reset_index(drop=True),
        small_frame["SOIL_MOISTURE_5"].reset_index(drop=True),
    )
    assert audit.loc[0, "SOIL_MOISTURE_5_attacked"] == pytest.approx(0.99)
    assert audit.loc[0, "is_attacked"]
    assert audit.loc[0, "attack_type"] == "bias"
    assert len(audit) == len(small_frame)


def test_build_detector_input_excludes_privileged_columns(small_frame, partition_plan) -> None:
    from attack_engine.dataset.schema import uscrn_hourly_schema

    schema = uscrn_hourly_schema()
    summary = _summary(())
    attacked = build_attacked_target(small_frame, schema, summary)
    partitions = assign_row_partitions(small_frame, schema, partition_plan)
    context_columns = ("T_HR_AVG", "SOIL_MOISTURE_10")

    detector_input = build_detector_input(
        small_frame, schema, attacked, partitions, context_columns
    )

    assert set(detector_input.columns) == {
        "station",
        "time",
        "partition",
        "SOIL_MOISTURE_5",
        "T_HR_AVG",
        "SOIL_MOISTURE_10",
    }
    assert "flag_incoherence_verticale" not in detector_input.columns
    assert "is_attacked" not in detector_input.columns
