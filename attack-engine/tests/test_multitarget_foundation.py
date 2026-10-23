"""Multi-target foundation: typed models, cell/row accounting, overlap and contracts.

These tests exercise the multi-variable-ready interfaces without any multi-variable
attack generation: every current transformation still modifies a single target column.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from attack_engine.attacks.config import SpikeParameters
from attack_engine.attacks.executor import AttackExecutor, ExecutionError
from attack_engine.attacks.models import (
    DEFAULT_TARGET_COLUMN,
    AttackDiagnostics,
    AttackRequest,
    AttackResult,
    AttackSettings,
    AttackStatus,
    ColumnModification,
)
from attack_engine.attacks.spike import apply_spike
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.exporting.builder import (
    build_attacked_target,
    build_cell_ground_truth,
    build_ground_truth,
)
from attack_engine.exporting.contracts import (
    VARIABLE_GROUND_TRUTH_COLUMNS,
    build_dataset_contract,
)
from attack_engine.exporting.models import ExportConfiguration, GroundTruthCellRecord
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler

RequestFactory = Callable[..., AttackRequest]
SETTINGS = AttackSettings()


def _single_column_result(labels, clean, candidate, changed) -> AttackResult:
    return AttackResult(
        window_id="w0",
        family=AttackFamily.SPIKE,
        station="AZ_Alpha_1_N",
        partition="validation",
        status=AttackStatus.APPLIED,
        row_labels=tuple(labels),
        clean_values=tuple(clean),
        candidate_values=tuple(candidate),
        changed_mask=tuple(changed),
        diagnostics=AttackDiagnostics(
            requested_magnitude=0.2,
            sign=1,
            modified_count=sum(changed),
            mean_abs_delta=0.0,
            max_abs_delta=0.0,
            saturated_count=0,
        ),
    )


# --- Request / result default to the single primary target ------------------------------


def test_request_defaults_to_primary_target(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.SPIKE, [0.3, 0.3, 0.3])
    assert request.resolved_target_columns == (DEFAULT_TARGET_COLUMN,)
    assert request.primary_target_column == DEFAULT_TARGET_COLUMN


def test_result_carries_target_columns(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.SPIKE, [0.3, 0.3, 0.3])
    result = apply_spike(
        request, SpikeParameters(magnitude=0.2), np.random.default_rng(0), SETTINGS
    )
    assert result.resolved_target_columns == (DEFAULT_TARGET_COLUMN,)
    assert len(result.columns) == 1
    assert result.columns[0].column == DEFAULT_TARGET_COLUMN


# --- Cells vs rows -----------------------------------------------------------------------


def test_single_target_cells_equal_rows_equal_modified_count() -> None:
    result = _single_column_result(
        [10, 11, 12], [0.2, 0.2, 0.2], [0.5, 0.2, 0.5], [True, False, True]
    )
    assert result.modified_count == 2
    assert result.modified_cells == 2
    assert result.modified_rows == 2
    assert result.modified_cell_keys() == ((10, DEFAULT_TARGET_COLUMN), (12, DEFAULT_TARGET_COLUMN))


def test_two_variable_result_distinguishes_cells_from_rows() -> None:
    # Hand-built two-variable result (no generator does this yet) to prove the accounting.
    result = _single_column_result([10, 11], [0.2, 0.2], [0.5, 0.2], [True, False])
    extra = ColumnModification(
        column="SOIL_MOISTURE_10",
        row_labels=(10, 11),
        clean_values=(0.3, 0.3),
        candidate_values=(0.3, 0.6),
        changed_mask=(False, True),
    )
    result = replace(result, extra_columns=(extra,))

    # row 10 changed in the primary variable, row 11 changed in the second variable.
    assert result.modified_cells == 2
    assert result.modified_rows == 2
    assert set(result.modified_cell_keys()) == {
        (10, DEFAULT_TARGET_COLUMN),
        (11, "SOIL_MOISTURE_10"),
    }


def test_same_row_two_variables_is_two_cells_one_row() -> None:
    result = _single_column_result([10], [0.2], [0.5], [True])
    extra = ColumnModification(
        column="SOIL_MOISTURE_10",
        row_labels=(10,),
        clean_values=(0.3,),
        candidate_values=(0.6,),
        changed_mask=(True,),
    )
    result = replace(result, extra_columns=(extra,))
    assert result.modified_cells == 2
    assert result.modified_rows == 1


# --- Overlap detection on (station, time, variable) == (row, column) --------------------


def test_executor_cell_overlap_guard_keys_on_row_and_column() -> None:
    # Two committed results writing the same row+column must raise.
    written: set = set()
    r1 = _single_column_result([10, 11], [0.2, 0.2], [0.5, 0.5], [True, True])
    r2 = _single_column_result([11, 12], [0.2, 0.2], [0.5, 0.5], [True, True])
    AttackExecutor._record_written(r1, written)
    with pytest.raises(ExecutionError, match="row 11"):
        AttackExecutor._record_written(r2, written)


def test_different_variables_same_row_do_not_collide() -> None:
    written: set = set()
    primary = _single_column_result([10], [0.2], [0.5], [True])
    other = _single_column_result([10], [0.2], [0.5], [True])
    other = replace(other, target_columns=("SOIL_MOISTURE_10",))
    AttackExecutor._record_written(primary, written)
    AttackExecutor._record_written(other, written)  # must not raise
    assert len(written) == 2


# --- Executor summary rows/cells reconcile for the single-target run ---------------------


def test_summary_cells_and_rows_equal_committed_for_single_target(
    schema: DatasetSchema,
    partition_plan: PartitionPlan,
    make_observations,
    make_scheduling_config,
    make_attack_config,
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(attack_budget=0.3, families=(AttackFamily.SPIKE,))
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    config = make_attack_config(spike={"magnitude": 0.2})
    summary = AttackExecutor(schema, partition_plan, config).execute(frame, plan)

    assert summary.committed_cells == summary.committed_total
    assert summary.committed_rows == summary.committed_total


# --- Export contracts --------------------------------------------------------------------


def test_contract_exposes_multitarget_fields(
    schema: DatasetSchema, partition_plan: PartitionPlan
) -> None:
    contract = build_dataset_contract(
        schema, partition_plan, ExportConfiguration(output_root="out"), ("P_CALC",)
    )
    assert contract.target_columns == (schema.target_column,)
    assert contract.attacked_columns == {schema.target_column: contract.attacked_target_column}
    assert contract.variable_ground_truth_columns == VARIABLE_GROUND_TRUTH_COLUMNS


def test_single_target_contract_fields_are_unchanged(
    schema: DatasetSchema, partition_plan: PartitionPlan
) -> None:
    contract = build_dataset_contract(
        schema, partition_plan, ExportConfiguration(output_root="out"), ("P_CALC",)
    )
    # Backward compatibility: the row-level ground-truth schema is untouched.
    assert contract.ground_truth_columns[0:2] == ("station", "time")
    assert "SOIL_MOISTURE_5_attacked" in contract.audit_extra_columns


def test_cell_ground_truth_matches_row_level_for_single_target(
    schema: DatasetSchema,
    partition_plan: PartitionPlan,
    make_observations,
    make_scheduling_config,
    make_attack_config,
) -> None:
    frame = make_observations(hours=336)
    scheduling = make_scheduling_config(attack_budget=0.3, families=(AttackFamily.SPIKE,))
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    config = make_attack_config(spike={"magnitude": 0.2})
    summary = AttackExecutor(schema, partition_plan, config).execute(frame, plan)

    attacked = build_attacked_target(frame, schema, summary)
    partition_labels = frame[schema.timestamps.column].map(lambda _: "validation")
    ground_truth = build_ground_truth(frame, schema, attacked, partition_labels)
    cells = build_cell_ground_truth(ground_truth, schema.target_column)

    assert tuple(cells.columns) == VARIABLE_GROUND_TRUTH_COLUMNS
    # One attacked cell per attacked row, all on the single target column.
    assert len(cells) == int(ground_truth["is_attacked"].sum()) == summary.committed_total
    assert set(cells["column"].unique()) == {schema.target_column}
    # Every cell row validates against the documented per-variable schema.
    for record in cells.head(20).to_dict("records"):
        GroundTruthCellRecord(**{**record, "time": pd.Timestamp(record["time"], tz="UTC")})
