from __future__ import annotations

import numpy as np
import pytest

from attack_engine.attacks import base
from attack_engine.attacks.models import AttackSettings, SignPolicy, TargetBounds


def test_quantize_snaps_to_grid() -> None:
    values = np.array([0.1234, 0.1255, 0.0])
    result = base.quantize(values, 0.001)
    np.testing.assert_allclose(result, [0.123, 0.126, 0.0])


def test_quantize_none_is_identity() -> None:
    values = np.array([0.123456])
    assert base.quantize(values, None) is values


def test_quantize_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        base.quantize(np.array([0.1]), 0.0)


def test_clip_reports_saturation() -> None:
    clipped, saturated = base.clip_to_bounds(np.array([-0.1, 0.5, 0.8]), TargetBounds(0.0, 0.70))
    np.testing.assert_allclose(clipped, [0.0, 0.5, 0.70])
    assert saturated.tolist() == [True, False, True]


@pytest.mark.parametrize(
    ("policy", "expected"), [(SignPolicy.POSITIVE, 1), (SignPolicy.NEGATIVE, -1)]
)
def test_resolve_sign_fixed(policy: SignPolicy, expected: int) -> None:
    assert base.resolve_sign(policy, np.random.default_rng(0)) == expected


def test_resolve_sign_random_is_deterministic() -> None:
    a = base.resolve_sign(SignPolicy.RANDOM, np.random.default_rng(1))
    b = base.resolve_sign(SignPolicy.RANDOM, np.random.default_rng(1))
    assert a == b
    assert a in (-1, 1)


def test_finalize_changed_mask_only_counts_real_changes() -> None:
    clean = np.array([0.20, 0.30, 0.40])
    candidate = np.array([0.25, 0.30, 0.45])  # middle unchanged
    settings = AttackSettings(bounds=TargetBounds(0.0, 0.70), quantum=None)
    final, changed, saturated = base.finalize_candidate(clean, candidate, settings)

    assert changed.tolist() == [True, False, True]
    assert saturated.tolist() == [False, False, False]
    assert final[1] == clean[1]


def test_finalize_zero_effect_after_quantization() -> None:
    clean = np.array([0.200, 0.300])
    candidate = np.array([0.2002, 0.3003])  # within half a 0.001 quantum of clean
    settings = AttackSettings(quantum=0.001)
    _, changed, _ = base.finalize_candidate(clean, candidate, settings)
    assert not changed.any()


def test_finalize_marks_saturation_only_when_changed() -> None:
    clean = np.array([0.69, 0.70])
    candidate = np.array([0.95, 0.95])  # second clips back to clean 0.70 -> not changed
    settings = AttackSettings(bounds=TargetBounds(0.0, 0.70))
    final, changed, saturated = base.finalize_candidate(clean, candidate, settings)
    assert changed.tolist() == [True, False]
    assert saturated.tolist() == [True, False]
    assert final.tolist() == [0.70, 0.70]
