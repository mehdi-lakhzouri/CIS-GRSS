from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from attack_engine.attacks.config import DriftParameters
from attack_engine.attacks.drift import apply_drift
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackSettings,
    SignPolicy,
    TargetBounds,
)
from attack_engine.scheduling.models import AttackFamily

RequestFactory = Callable[..., AttackRequest]
SETTINGS = AttackSettings(bounds=TargetBounds(0.0, 0.70), quantum=None)


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_drift_ramp_is_linear_in_elapsed_time(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.DRIFT, [0.30] * 5)
    params = DriftParameters(final_magnitude=0.1, sign=SignPolicy.POSITIVE, hold_first=True)

    result = apply_drift(request, params, _rng(), SETTINGS)

    deltas = [c - x for x, c in zip(result.clean_values, result.candidate_values, strict=True)]
    assert deltas[0] == pytest.approx(0.0)  # first point held
    # elapsed hours 0,1,2,3,4 -> ramp 0, .025, .05, .075, .1
    assert deltas[-1] == pytest.approx(0.1)
    assert deltas[2] == pytest.approx(0.05)


def test_drift_first_point_unchanged_reflected_in_mask(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.DRIFT, [0.30] * 5)
    result = apply_drift(
        request, DriftParameters(final_magnitude=0.1, sign=SignPolicy.POSITIVE), _rng(), SETTINGS
    )
    assert result.changed_mask[0] is False
    assert result.modified_count == 4


def test_drift_negative(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.DRIFT, [0.30] * 4)
    result = apply_drift(
        request, DriftParameters(final_magnitude=0.1, sign=SignPolicy.NEGATIVE), _rng(), SETTINGS
    )
    assert result.candidate_values[-1] == pytest.approx(0.20)


def test_drift_rejects_window_too_short(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.DRIFT, [0.30])
    result = apply_drift(request, DriftParameters(final_magnitude=0.1), _rng(), SETTINGS)
    assert result.rejection_reason is AttackRejectionReason.WINDOW_TOO_SHORT


def test_drift_rejects_excessive_slope(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.DRIFT, [0.30] * 3)
    # slope = 0.1 / 2 h = 0.05/h > max 0.01/h
    params = DriftParameters(final_magnitude=0.1, max_slope_per_hour=0.01)
    result = apply_drift(request, params, _rng(), SETTINGS)
    assert result.rejection_reason is AttackRejectionReason.SLOPE_TOO_STEEP


def test_drift_slope_within_limit_is_applied(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.DRIFT, [0.30] * 11)
    # slope = 0.1 / 10 h = 0.01/h == max
    params = DriftParameters(final_magnitude=0.1, max_slope_per_hour=0.01, sign=SignPolicy.POSITIVE)
    result = apply_drift(request, params, _rng(), SETTINGS)
    assert result.applied
    assert result.diagnostics.extra["slope_per_hour"] == pytest.approx(0.01)


def test_drift_clips_to_bounds(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.DRIFT, [0.68] * 5)
    params = DriftParameters(final_magnitude=0.1, sign=SignPolicy.POSITIVE)
    result = apply_drift(request, params, _rng(), SETTINGS)
    assert all(value <= 0.70 for value in result.candidate_values)
    assert result.diagnostics.saturated_count >= 1


def test_drift_is_deterministic(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.DRIFT, [0.30] * 6)
    params = DriftParameters(final_magnitude=0.1, sign=SignPolicy.RANDOM)
    first = apply_drift(request, params, np.random.default_rng(4), SETTINGS)
    second = apply_drift(request, params, np.random.default_rng(4), SETTINGS)
    assert first.candidate_values == second.candidate_values
