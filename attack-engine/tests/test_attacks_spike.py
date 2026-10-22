from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from attack_engine.attacks.config import SpikeParameters
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackSettings,
    SignPolicy,
    TargetBounds,
)
from attack_engine.attacks.spike import apply_spike
from attack_engine.scheduling.models import AttackFamily

RequestFactory = Callable[..., AttackRequest]
SETTINGS = AttackSettings(bounds=TargetBounds(0.0, 0.70), quantum=None)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_spike_modifies_between_one_and_three_points(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.SPIKE, [0.3] * 6)
    params = SpikeParameters(magnitude=0.2, sign=SignPolicy.POSITIVE, min_points=1, max_points=3)

    result = apply_spike(request, params, _rng(), SETTINGS)

    assert result.applied
    assert 1 <= result.modified_count <= 3
    assert result.modified_count == sum(result.changed_mask)


def test_spike_positive_increases_values(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.SPIKE, [0.3] * 5)
    params = SpikeParameters(magnitude=0.2, sign=SignPolicy.POSITIVE, min_points=2, max_points=2)
    result = apply_spike(request, params, _rng(3), SETTINGS)

    for clean, cand, changed in zip(
        result.clean_values, result.candidate_values, result.changed_mask, strict=True
    ):
        if changed:
            assert cand == pytest.approx(clean + 0.2)


def test_spike_negative_decreases_values(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.SPIKE, [0.3] * 5)
    params = SpikeParameters(magnitude=0.2, sign=SignPolicy.NEGATIVE, min_points=2, max_points=2)
    result = apply_spike(request, params, _rng(3), SETTINGS)

    deltas = [
        c - x
        for x, c, m in zip(
            result.clean_values, result.candidate_values, result.changed_mask, strict=True
        )
        if m
    ]
    assert deltas and all(delta == pytest.approx(-0.2) for delta in deltas)


def test_spike_clips_to_bounds(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.SPIKE, [0.65] * 4)
    params = SpikeParameters(magnitude=0.5, sign=SignPolicy.POSITIVE, min_points=4, max_points=4)
    result = apply_spike(request, params, _rng(), SETTINGS)

    assert all(value <= 0.70 for value in result.candidate_values)
    assert result.diagnostics.saturated_count >= 1


def test_spike_count_not_inferred_from_window_length(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.SPIKE, [0.3] * 20)
    params = SpikeParameters(magnitude=0.2, min_points=1, max_points=3)
    result = apply_spike(request, params, _rng(), SETTINGS)
    assert result.modified_count <= 3  # never the full 20-row window


def test_spike_positions_without_replacement(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.SPIKE, [0.3] * 3)
    params = SpikeParameters(magnitude=0.2, min_points=3, max_points=3)
    result = apply_spike(request, params, _rng(), SETTINGS)
    positions = result.diagnostics.extra["positions"].split(",")
    assert len(positions) == len(set(positions)) == 3


def test_spike_is_deterministic(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.SPIKE, [0.3] * 6)
    params = SpikeParameters(magnitude=0.2, min_points=1, max_points=3)
    first = apply_spike(request, params, _rng(5), SETTINGS)
    second = apply_spike(request, params, _rng(5), SETTINGS)
    assert first.candidate_values == second.candidate_values
    assert first.changed_mask == second.changed_mask


def test_spike_rejects_missing_target(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.SPIKE, [0.3, np.nan, 0.3])
    params = SpikeParameters(magnitude=0.2)
    result = apply_spike(request, params, _rng(), SETTINGS)
    assert not result.applied
    assert result.rejection_reason is AttackRejectionReason.MISSING_TARGET


def test_spike_zero_effect_when_already_at_bound(make_attack_request: RequestFactory) -> None:
    # Clean already at the upper bound; a positive spike clips back to the same value.
    request = make_attack_request(AttackFamily.SPIKE, [0.70] * 3)
    params = SpikeParameters(magnitude=0.2, sign=SignPolicy.POSITIVE, min_points=3, max_points=3)
    result = apply_spike(request, params, _rng(), SETTINGS)
    assert not result.applied
    assert result.rejection_reason is AttackRejectionReason.ZERO_EFFECT


def test_spike_does_not_mutate_request(make_attack_request: RequestFactory) -> None:
    clean = [0.3] * 5
    request = make_attack_request(AttackFamily.SPIKE, clean)
    apply_spike(request, SpikeParameters(magnitude=0.2), _rng(), SETTINGS)
    assert request.clean_values == tuple(clean)
