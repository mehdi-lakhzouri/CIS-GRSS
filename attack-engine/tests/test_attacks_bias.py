from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from attack_engine.attacks.bias import apply_bias
from attack_engine.attacks.config import BiasParameters
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


def test_bias_applies_constant_offset(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.BIAS, [0.20, 0.30, 0.40, 0.25])
    result = apply_bias(
        request, BiasParameters(magnitude=0.1, sign=SignPolicy.POSITIVE), _rng(), SETTINGS
    )

    assert result.applied
    assert result.modified_count == 4
    deltas = [c - x for x, c in zip(result.clean_values, result.candidate_values, strict=True)]
    assert all(delta == pytest.approx(0.1) for delta in deltas)
    assert result.diagnostics.extra["realized_offset"] == pytest.approx(0.1)


def test_bias_negative(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.BIAS, [0.30, 0.40])
    result = apply_bias(
        request, BiasParameters(magnitude=0.1, sign=SignPolicy.NEGATIVE), _rng(), SETTINGS
    )
    deltas = [c - x for x, c in zip(result.clean_values, result.candidate_values, strict=True)]
    assert all(delta == pytest.approx(-0.1) for delta in deltas)


def test_bias_rejects_infeasible_offset(make_attack_request: RequestFactory) -> None:
    # 0.65 + 0.1 = 0.75 exceeds the upper bound -> reject rather than silently clip.
    request = make_attack_request(AttackFamily.BIAS, [0.65, 0.30])
    result = apply_bias(
        request, BiasParameters(magnitude=0.1, sign=SignPolicy.POSITIVE), _rng(), SETTINGS
    )

    assert not result.applied
    assert result.rejection_reason is AttackRejectionReason.INFEASIBLE_BOUNDS
    assert result.candidate_values == result.clean_values


def test_bias_offset_stays_constant_with_quantization(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.BIAS, [0.201, 0.302, 0.403])
    settings = AttackSettings(bounds=TargetBounds(0.0, 0.70), quantum=0.001)
    result = apply_bias(
        request, BiasParameters(magnitude=0.05, sign=SignPolicy.POSITIVE), _rng(), settings
    )

    assert result.applied
    deltas = [
        round(c - x, 6) for x, c in zip(result.clean_values, result.candidate_values, strict=True)
    ]
    assert len(set(deltas)) == 1


def test_bias_rejects_missing_target(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.BIAS, [0.3, np.nan])
    result = apply_bias(request, BiasParameters(magnitude=0.1), _rng(), SETTINGS)
    assert result.rejection_reason is AttackRejectionReason.MISSING_TARGET


def test_bias_is_deterministic(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.BIAS, [0.2, 0.3, 0.4])
    params = BiasParameters(magnitude=0.1, sign=SignPolicy.RANDOM)
    first = apply_bias(request, params, np.random.default_rng(2), SETTINGS)
    second = apply_bias(request, params, np.random.default_rng(2), SETTINGS)
    assert first.candidate_values == second.candidate_values
