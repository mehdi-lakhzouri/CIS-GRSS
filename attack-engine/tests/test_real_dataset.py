"""Read-only checks against the real reference delivery.

Opt-in: set ATTACK_ENGINE_REAL_DATASET to the path of uscrn_clean.parquet. The
expected figures below are those of the known reference delivery.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attack_engine.attacks.calibration import feature_vector, fit_calibration, load_calibration
from attack_engine.attacks.config import load_attack_config
from attack_engine.attacks.context import StealthContextProvider
from attack_engine.attacks.executor import AttackExecutor
from attack_engine.config import load_config
from attack_engine.dataset.loader import DatasetLoader
from attack_engine.dataset.profiler import DatasetProfiler
from attack_engine.dataset.schema import uscrn_hourly_schema
from attack_engine.dataset.validator import DatasetValidator
from attack_engine.exporting.builder import build_cell_ground_truth
from attack_engine.exporting.contracts import VARIABLE_GROUND_TRUTH_COLUMNS
from attack_engine.exporting.models import ExportConfiguration
from attack_engine.exporting.pipeline import export_run
from attack_engine.scheduling.budget import ReservationStatus
from attack_engine.scheduling.config import load_scheduling_config
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler
from attack_engine.utils.hashing import IntegrityStatus, compare_sha256, sha256_file

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PROJECT_CONFIG = _CONFIG_DIR / "dataset.yaml"
SCHEDULING_CONFIG = _CONFIG_DIR / "scheduling.yaml"
ATTACK_PARAMS = _CONFIG_DIR / "attacks.yaml"
_REAL_PATH = os.environ.get("ATTACK_ENGINE_REAL_DATASET")

pytestmark = [
    pytest.mark.real_dataset,
    pytest.mark.skipif(not _REAL_PATH, reason="ATTACK_ENGINE_REAL_DATASET is not set"),
]


def test_reference_delivery_validates_and_is_unchanged() -> None:
    path = Path(str(_REAL_PATH))
    config = load_config(PROJECT_CONFIG)
    schema = uscrn_hourly_schema()
    digest_before = sha256_file(path)

    dataset = DatasetLoader(schema).load(path)
    report = DatasetValidator(schema, config.validation_profile, config.partitions).validate(
        dataset.frame
    )
    profile = DatasetProfiler(schema).profile(dataset.frame, report)

    assert report.passed, report.errors
    assert compare_sha256(dataset.source_sha256, config.expected_sha256).status is (
        IntegrityStatus.MATCH
    )
    assert report.partition_row_counts == {"train": 105120, "validation": 52704, "test": 52560}
    assert profile.target.missing_count == 5401
    assert profile.year_boundary_row_count == 24
    assert sha256_file(path) == digest_before


def test_reference_delivery_plan_is_reproducible_and_read_only() -> None:
    path = Path(str(_REAL_PATH))
    config = load_config(PROJECT_CONFIG)
    scheduling = load_scheduling_config(SCHEDULING_CONFIG)
    schema = uscrn_hourly_schema()
    digest_before = sha256_file(path)

    dataset = DatasetLoader(schema).load(path)
    snapshot = dataset.frame.copy(deep=True)
    scheduler = AttackWindowScheduler(schema, config.partitions, scheduling)
    first = scheduler.plan(dataset.frame)
    second = scheduler.plan(dataset.frame)

    # Budget denominator is the finite-target count over the scope.
    assert first.requested_quota > 0
    assert first.reserved_total <= first.requested_quota
    assert first.ledger.total_committed == 0
    signature = [(w.window_id, w.first_observation_time, w.row_labels) for w in first.windows]
    assert signature == [
        (w.window_id, w.first_observation_time, w.row_labels) for w in second.windows
    ]
    known_partitions = {partition.name for partition in config.partitions.partitions}
    assert all(window.partition in known_partitions for window in first.windows)

    pd.testing.assert_frame_equal(dataset.frame, snapshot, check_exact=True)
    assert sha256_file(path) == digest_before


def test_reference_delivery_execution_is_reproducible_and_read_only() -> None:
    path = Path(str(_REAL_PATH))
    config = load_config(PROJECT_CONFIG)
    scheduling = load_scheduling_config(SCHEDULING_CONFIG)
    attack_config = load_attack_config(ATTACK_PARAMS)
    schema = uscrn_hourly_schema()
    digest_before = sha256_file(path)

    dataset = DatasetLoader(schema).load(path)
    snapshot = dataset.frame.copy(deep=True)
    executor = AttackExecutor(schema, config.partitions, attack_config)

    plan_a = AttackWindowScheduler(schema, config.partitions, scheduling).plan(dataset.frame)
    summary_a = executor.execute(dataset.frame, plan_a)
    plan_b = AttackWindowScheduler(schema, config.partitions, scheduling).plan(dataset.frame)
    summary_b = executor.execute(dataset.frame, plan_b)

    # Reserved capacity is never reported as attacked; commits are the real counts.
    assert summary_a.committed_total <= summary_a.reserved_before <= summary_a.requested_quota
    assert summary_a.committed_total == plan_a.ledger.total_committed
    assert plan_a.ledger.total_reserved == plan_a.ledger.total_committed
    assert all(cell.active_reservation_count == 0 for cell in plan_a.ledger.cell_ledgers())
    assert summary_a.frame_unchanged
    assert not summary_a.skipped_families
    stealth = summary_a.per_family[AttackFamily.STEALTH]
    assert stealth.planned_windows == 49
    assert stealth.applied_windows > 0
    assert stealth.committed > 0
    assert summary_a.calibration_sha256 is not None
    assert summary_a.context_available <= summary_a.context_total == stealth.reserved
    # Same dataset, seeds and configuration reproduce identical results.
    assert plan_a.windows == plan_b.windows
    assert summary_a == summary_b

    windows = {window.window_id: window for window in plan_a.windows}
    observed = snapshot[schema.target_column].copy()
    modified_labels: set[int] = set()
    emitted: dict[int, float] = {}
    for result in summary_a.results:
        clean = np.asarray(result.clean_values)
        final = np.asarray(result.candidate_values)
        actual_mask = np.isfinite(clean) & np.isfinite(final) & (final != clean)
        np.testing.assert_array_equal(actual_mask, result.changed_mask)
        assert result.modified_count == int(actual_mask.sum())
        assert np.isfinite(final).all()
        assert (attack_config.target_low <= final).all()
        assert (final <= attack_config.target_high).all()
        window = windows[result.window_id]
        np.testing.assert_array_equal(
            clean, snapshot.loc[list(window.row_labels), schema.target_column].to_numpy()
        )
        reservation = plan_a.ledger.reservation(result.window_id)
        if result.applied:
            assert reservation.status is ReservationStatus.COMMITTED
            assert reservation.committed_count == int(actual_mask.sum())
        else:
            assert reservation.status is ReservationStatus.CANCELLED
            assert not actual_mask.any()
        changed_labels = [
            label for label, changed in zip(result.row_labels, actual_mask, strict=True) if changed
        ]
        assert modified_labels.isdisjoint(changed_labels)
        modified_labels.update(changed_labels)
        emitted.update(
            {
                label: value
                for label, value, changed in zip(
                    result.row_labels, result.candidate_values, result.changed_mask, strict=True
                )
                if changed
            }
        )
        observed.loc[list(result.row_labels)] = final
        if result.family is AttackFamily.DRIFT and result.applied:
            assert len(clean) == window.length
            assert result.changed_mask[0] is False
            assert final[0] == clean[0]
            assert result.modified_count <= window.length - 1

    drift = summary_a.per_family[AttackFamily.DRIFT]
    assert drift.committed <= drift.reserved - drift.applied_windows
    # Fixed reference delivery and checked-in seed/configuration: the audited regression.
    assert (drift.applied_windows, drift.reserved, drift.committed) == (42, 2882, 2839)
    assert summary_a.committed_total == 8989 + stealth.committed == len(modified_labels)
    assert summary_a.committed_total == 9679
    assert stealth.committed == 690
    clean_target = snapshot[schema.target_column]
    final_mask = np.isfinite(clean_target) & np.isfinite(observed) & (observed != clean_target)
    assert int(final_mask.sum()) == summary_a.committed_total
    np.testing.assert_array_equal(observed.isna(), clean_target.isna())
    assert all(
        cell.remaining == cell.allocated - cell.committed for cell in plan_a.ledger.cell_ledgers()
    )

    calibration = load_calibration(attack_config.stealth.calibration_path)
    context_provider = StealthContextProvider(
        dataset.frame, schema, config.partitions, plan_a, calibration
    )
    params = attack_config.stealth
    for result in summary_a.applied_results:
        if result.family is not AttackFamily.STEALTH:
            continue
        assert result.partition in params.attack_partitions
        context = context_provider.for_window(windows[result.window_id], emitted)
        model = calibration.station_models[result.station]
        history = context.history_values
        previous_time = context.history_times[-1]
        previous_value = history[-1]
        previous_offset = cumulative = 0.0
        for i, value in enumerate(result.candidate_values):
            dt = (context.times[i] - previous_time).total_seconds() / 3600
            mu = model.predict(
                feature_vector(
                    tuple(
                        context.measurements[name][i] for name in params.required_context_features
                    ),
                    history,
                    context.times[i],
                )
            )
            assert (
                abs(value - mu)
                <= model.residual_tolerance * params.residual_tolerance_multiplier + 1e-15
            )
            assert abs(value - previous_value) / dt <= params.max_step_change + 1e-15
            offset = value - result.clean_values[i]
            assert abs(offset - previous_offset) / dt <= params.max_change_rate + 1e-15
            cumulative += abs(offset - previous_offset)
            assert cumulative <= params.max_cumulative_drift + 1e-15
            history = (*history[1:], value)
            previous_time, previous_value, previous_offset = context.times[i], value, offset
        assert np.mean(np.abs(np.asarray(result.candidate_values) - result.clean_values)) >= (
            params.min_attack_impact
        )

    pd.testing.assert_frame_equal(dataset.frame, snapshot, check_exact=True)
    assert sha256_file(path) == digest_before


def test_reference_stealth_calibration_ignores_all_future_measurements() -> None:
    dataset_config = load_config(PROJECT_CONFIG)
    params = load_attack_config(ATTACK_PARAMS).stealth
    schema = uscrn_hourly_schema()
    dataset = DatasetLoader(schema).load(Path(str(_REAL_PATH)))
    first = fit_calibration(dataset.frame, schema, dataset_config.partitions, params)
    modified = dataset.frame.copy(deep=True)
    modified.loc[
        modified.annee.ge(2024), [schema.target_column, *params.required_context_features]
    ] = -99999.0
    second = fit_calibration(modified, schema, dataset_config.partitions, params)
    assert first == second == load_calibration(params.calibration_path)
    assert sha256_file(Path(str(_REAL_PATH))) == dataset.source_sha256


def test_reference_delivery_export_reconciles_with_execution(tmp_path: Path) -> None:
    """Phase 5: the full export bundle on the real reference delivery, read back."""
    path = Path(str(_REAL_PATH))
    config = load_config(PROJECT_CONFIG)
    scheduling = load_scheduling_config(SCHEDULING_CONFIG)
    attack_config = load_attack_config(ATTACK_PARAMS)
    schema = uscrn_hourly_schema()
    digest_before = sha256_file(path)

    dataset = DatasetLoader(schema).load(path)
    plan = AttackWindowScheduler(schema, config.partitions, scheduling).plan(dataset.frame)
    summary = AttackExecutor(schema, config.partitions, attack_config).execute(dataset.frame, plan)

    export_config = ExportConfiguration(
        output_root=tmp_path / "generated", run_id="phase5-real-check"
    )
    result = export_run(
        dataset=dataset,
        schema=schema,
        partitions=config.partitions,
        scheduling_config_path=SCHEDULING_CONFIG,
        attack_config=attack_config,
        attack_params_path=ATTACK_PARAMS,
        plan=plan,
        summary=summary,
        config=export_config,
    )

    assert result.row_count == dataset.metadata.row_count == 210384
    assert result.missing_target_count == 5401
    assert result.modified_count == summary.committed_total == 9679
    # Multi-target foundation: the single-target run modifies one cell per attacked row, so
    # cells and distinct rows both equal the committed total; the stealth family contributes.
    assert summary.committed_cells == summary.committed_rows == 9679
    assert summary.per_family[AttackFamily.STEALTH].committed > 0
    assert result.validation_passed, result.validation_messages

    ground_truth = pd.read_parquet(result.run_dir / "ground_truth.parquet")
    manifest = pd.read_csv(result.run_dir / "attack_manifest.csv")
    detector_input = pd.read_parquet(result.run_dir / "detector_input.parquet")
    audit = pd.read_parquet(result.run_dir / "uscrn_attacked.parquet")

    assert int(ground_truth["is_attacked"].sum()) == 9679
    assert int((~ground_truth["is_evaluable"]).sum()) == 5401
    assert (
        int(manifest.loc[manifest["status"] == "applied", "modified_count"].sum())
        == summary.committed_total
    )
    assert not (set(detector_input.columns) & {"is_attacked", "attack_type", "attack_id"})
    assert "flag_incoherence_verticale" not in detector_input.columns
    for column in schema.column_names:
        pd.testing.assert_series_equal(
            audit[column].reset_index(drop=True), dataset.frame[column].reset_index(drop=True)
        )

    # Multi-target foundation: variable-level ground truth projects the single target.
    cells = build_cell_ground_truth(ground_truth, schema.target_column)
    assert tuple(cells.columns) == VARIABLE_GROUND_TRUTH_COLUMNS
    assert len(cells) == 9679
    assert set(cells["column"].unique()) == {schema.target_column}

    assert sha256_file(path) == digest_before == dataset.source_sha256


@pytest.mark.parametrize("family", tuple(AttackFamily))
def test_all_multivariable_families_on_real_dataset_are_read_only_and_deterministic(family) -> None:
    path = Path(str(_REAL_PATH))
    config = load_config(PROJECT_CONFIG)
    schema = uscrn_hourly_schema()
    campaigns = _CONFIG_DIR / "campaigns" / family.value
    scheduling = load_scheduling_config(campaigns / "scheduling.yaml")
    attack_config = load_attack_config(campaigns / "attacks.yaml")
    digest_before = sha256_file(path)

    dataset = DatasetLoader(schema).load(path)
    snapshot = dataset.frame.copy(deep=True)
    scheduler = AttackWindowScheduler(schema, config.partitions, scheduling)
    executor = AttackExecutor(schema, config.partitions, attack_config)
    first = executor.execute(dataset.frame, scheduler.plan(dataset.frame))
    second = executor.execute(dataset.frame, scheduler.plan(dataset.frame))

    family_summary = first.per_family[family]
    assert family_summary.applied_windows > 0
    assert first.committed_total == first.committed_cells > first.committed_rows
    assert first.committed_cells <= first.reserved_before <= first.requested_quota
    assert first.committed_cells <= scheduling.max_modified_cells
    assert first.committed_rows <= scheduling.max_modified_rows
    assert family_summary.committed == first.committed_total
    assert sum(first.per_variable_modifications.values()) == first.committed_cells
    assert first.per_variable_modifications == second.per_variable_modifications
    assert first.skipped_families == ()
    assert len(first.results) == len(second.results)
    written = set()
    for result, repeated in zip(first.results, second.results, strict=True):
        assert result.window_id == repeated.window_id
        assert result.status == repeated.status
        assert result.rejection_reason == repeated.rejection_reason
        assert result.partition in ("validation", "test")
        assert result.changed_variables == (
            result.resolved_target_columns if result.applied else ()
        )
        assert written.isdisjoint(result.modified_cell_keys())
        written.update(result.modified_cell_keys())
        for column, repeated_column in zip(result.columns, repeated.columns, strict=True):
            clean, final = np.asarray(column.clean_values), np.asarray(column.candidate_values)
            np.testing.assert_array_equal(final, repeated_column.candidate_values)
            np.testing.assert_array_equal(
                clean, snapshot.loc[list(column.row_labels), column.column]
            )
            mask = np.isfinite(clean) & np.isfinite(final) & (clean != final)
            np.testing.assert_array_equal(mask, column.changed_mask)
            np.testing.assert_array_equal(np.isnan(clean), np.isnan(final))
            assert (final[mask] >= 0).all()
            assert (final[mask] <= (200 if column.column == "P_CALC" else 0.7)).all()
            if result.family is AttackFamily.REPLAY and result.applied:
                source = result.replay_source
                assert source.last_time < pd.Timestamp(
                    snapshot.loc[result.row_labels[0], "time"], tz="UTC"
                )
                source_values = snapshot.loc[list(source.row_labels), column.column].to_numpy()
                np.testing.assert_array_equal(
                    final[np.isfinite(clean)], source_values[np.isfinite(clean)]
                )
    assert len(written) == first.committed_cells

    pd.testing.assert_frame_equal(dataset.frame, snapshot, check_exact=True)
    assert sha256_file(path) == digest_before == dataset.source_sha256
