"""Coordinated contextual stealth (a strategy of the stealth family): calibration,
secondary physics and executor integration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from attack_engine.attacks.config import StealthParameters
from attack_engine.attacks.executor import AttackExecutor
from attack_engine.attacks.models import (
    AttackSettings,
    MultiStealthContext,
    TargetBounds,
)
from attack_engine.attacks.multistealth import (
    MOISTURE_10,
    RAINFALL,
    SecondaryCalibration,
    StationCoupling,
    _depth_candidate,
    _rainfall_candidate,
    fit_secondary_calibration,
)
from attack_engine.attacks.stealth import apply_stealth
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler

SETTINGS = AttackSettings(bounds=TargetBounds(0.0, 0.70), quantum=0.001)


def _multi_params(stealth_parameters, secondary_columns) -> StealthParameters:
    return StealthParameters(
        **{
            **stealth_parameters.model_dump(),
            "secondary_columns": secondary_columns,
            "target_columns": ("SOIL_MOISTURE_5", *secondary_columns),
            "rainfall_trigger": 0.001,
        }
    )


def _station_calibration(station: str, *, slope: float, ceiling: float) -> SecondaryCalibration:
    return SecondaryCalibration(
        secondary_columns=(MOISTURE_10, RAINFALL),
        partition_start="2022-01-01T00:00:00+00:00",
        partition_end="2024-01-01T00:00:00+00:00",
        stations={
            station: StationCoupling(
                depth_slope=slope, depth_rows=500, rainfall_ceiling_mm=ceiling, rainfall_rows=50
            )
        },
    )


def _params_only(**overrides) -> StealthParameters:
    from pathlib import Path

    base = {
        "calibration_path": Path("x.json"),
        "secondary_columns": (MOISTURE_10, RAINFALL),
        "min_history_hours": 4,
        "min_training_rows": 24,
    }
    return StealthParameters(**{**base, **overrides})


# --- secondary candidate physics ---------------------------------------------------------


def test_depth_candidate_follows_lagged_damped_offset() -> None:
    clean = np.full(5, 0.30)
    offset5 = np.array([0.0, 0.02, 0.02, 0.02, 0.02])
    coupling = StationCoupling(
        depth_slope=0.5, depth_rows=1, rainfall_ceiling_mm=0.0, rainfall_rows=0
    )
    candidate = _depth_candidate(clean, offset5, coupling, _params_only())
    assert candidate[0] == pytest.approx(0.30)
    assert candidate[2] == pytest.approx(0.31)
    assert candidate.max() <= 0.70


def test_depth_candidate_preserves_missing() -> None:
    clean = np.array([0.30, np.nan, 0.30])
    coupling = StationCoupling(
        depth_slope=1.0, depth_rows=1, rainfall_ceiling_mm=0.0, rainfall_rows=0
    )
    candidate = _depth_candidate(clean, np.full(3, 0.02), coupling, _params_only())
    assert np.isnan(candidate[1])


def test_rainfall_pulse_precedes_rise_and_respects_ceiling() -> None:
    clean = np.zeros(6)
    offset5 = np.array([0.0, 0.0, 0.0, 0.02, 0.02, 0.02])
    coupling = StationCoupling(
        depth_slope=0.0, depth_rows=0, rainfall_ceiling_mm=0.4, rainfall_rows=10
    )
    params = _params_only(rainfall_lag_hours=2, rainfall_pulse_mm=1.0, rainfall_trigger=0.003)
    candidate = _rainfall_candidate(clean, offset5, coupling, params)
    assert candidate[1] == pytest.approx(0.4)  # capped at the ceiling, not 1.0
    assert candidate[0] == 0.0 and candidate[3] == 0.0


def test_rainfall_never_fills_missing_slot() -> None:
    clean = np.array([np.nan, np.nan, 0.0, 0.0, 0.0])
    offset5 = np.array([0.0, 0.0, 0.0, 0.02, 0.02])
    coupling = StationCoupling(
        depth_slope=0.0, depth_rows=0, rainfall_ceiling_mm=1.0, rainfall_rows=10
    )
    params = _params_only(rainfall_lag_hours=2, rainfall_pulse_mm=0.5)
    candidate = _rainfall_candidate(clean, offset5, coupling, params)
    assert np.isnan(candidate[1])


# --- calibration: train only, no leakage -------------------------------------------------


def test_secondary_calibration_is_train_only(
    stealth_frame, schema, partition_plan, stealth_parameters
) -> None:
    frame = stealth_frame.copy()
    frame["SOIL_MOISTURE_10"] = frame["SOIL_MOISTURE_5"]
    params = _multi_params(stealth_parameters, (MOISTURE_10, RAINFALL))

    slope_before = (
        fit_secondary_calibration(frame, schema, partition_plan, params)
        .coupling("AZ_Alpha_1_N")
        .depth_slope
    )
    assert slope_before > 0.5

    leak = frame.copy()
    leak.loc[leak["time"].dt.year.isin([2024, 2025]), "SOIL_MOISTURE_10"] = 0.0
    slope_after = (
        fit_secondary_calibration(leak, schema, partition_plan, params)
        .coupling("AZ_Alpha_1_N")
        .depth_slope
    )
    assert slope_after == pytest.approx(slope_before)


# --- coordinated stealth event via the stealth family ------------------------------------


def _coordinated_request(make_stealth_request, stealth_parameters, secondary, *, slope, ceiling):
    request = make_stealth_request(length=12, offset=24, partition="validation")
    station = request.window.station
    secondary_clean = {
        MOISTURE_10: tuple(0.30 for _ in request.window.row_labels),
        RAINFALL: tuple(0.0 for _ in request.window.row_labels),
    }
    request = replace(
        request,
        target_columns=("SOIL_MOISTURE_5", *secondary),
        multistealth_context=MultiStealthContext(
            secondary_clean={k: secondary_clean[k] for k in secondary},
            calibration=_station_calibration(station, slope=slope, ceiling=ceiling),
        ),
    )
    return request


def test_stealth_secondary_modifies_multiple_columns(
    make_stealth_request, stealth_parameters
) -> None:
    secondary = (MOISTURE_10, RAINFALL)
    request = _coordinated_request(
        make_stealth_request, stealth_parameters, secondary, slope=1.0, ceiling=5.0
    )
    params = _multi_params(stealth_parameters, secondary)

    result = apply_stealth(request, params, np.random.default_rng(0), SETTINGS)

    assert result.applied
    assert result.family is AttackFamily.STEALTH
    assert MOISTURE_10 in {c.column for c in result.columns}
    assert result.modified_count == result.primary_column.modified_count
    assert result.modified_cells >= result.modified_count
    keys = result.modified_cell_keys()
    assert len(keys) == len(set(keys))


def test_single_target_stealth_is_unchanged(make_stealth_request, stealth_parameters) -> None:
    request = make_stealth_request(length=12, offset=24, partition="validation")
    result = apply_stealth(request, stealth_parameters, np.random.default_rng(0), SETTINGS)
    # No secondary columns configured -> plain single-target stealth, no extra columns.
    assert result.extra_columns == ()
    assert result.modified_cells == result.modified_count


def test_coordinated_stealth_is_deterministic(make_stealth_request, stealth_parameters) -> None:
    def run():
        request = _coordinated_request(
            make_stealth_request, stealth_parameters, (MOISTURE_10,), slope=1.0, ceiling=0.0
        )
        params = _multi_params(stealth_parameters, (MOISTURE_10,))
        return apply_stealth(request, params, np.random.default_rng(1), SETTINGS)

    first, second = run(), run()
    assert first.candidate_values == second.candidate_values
    assert [c.candidate_values for c in first.extra_columns] == [
        c.candidate_values for c in second.extra_columns
    ]


# --- executor integration ----------------------------------------------------------------


def test_executor_coordinated_stealth_tracks_cells(
    stealth_frame,
    stealth_parameters,
    stealth_calibration,
    schema,
    partition_plan,
    make_scheduling_config,
    make_attack_config,
) -> None:
    frame = stealth_frame.copy()
    frame["SOIL_MOISTURE_10"] = frame["SOIL_MOISTURE_5"]  # strong train coupling
    params = _multi_params(stealth_parameters, (MOISTURE_10, RAINFALL))
    scheduling = make_scheduling_config(
        attack_budget=0.5,
        families=(AttackFamily.STEALTH,),
        budget_scope=("validation", "test"),
        min_window_length=12,
        max_window_length=24,
        min_history={AttackFamily.STEALTH: 8},
        variable_count={AttackFamily.STEALTH: 3},
    )
    plan = AttackWindowScheduler(schema, partition_plan, scheduling).plan(frame)
    config = make_attack_config(stealth=params.model_dump())
    summary = AttackExecutor(schema, partition_plan, config).execute(frame, plan)

    family = summary.per_family[AttackFamily.STEALTH]
    assert family.applied_windows > 0
    assert summary.committed_cells > summary.committed_rows  # depth coupling adds cells
    assert all(w.partition in ("validation", "test") for w in plan.windows)
    assert summary.frame_unchanged
    all_cells = [k for r in summary.applied_results for k in r.modified_cell_keys()]
    assert len(all_cells) == len(set(all_cells))  # no conflicting writes
