"""Spike injection: abrupt change at 1-3 positions of the window.

``k`` positions (``k`` drawn in ``[min_points, max_points]`` and capped at the
window length) are chosen without replacement and shifted by ``sign * magnitude``.
Values are clipped to the target bounds; a position that lands back on its clean
value after clipping and quantization is not counted as modified.
"""

from __future__ import annotations

import numpy as np

from attack_engine.attacks import base
from attack_engine.attacks.config import SpikeParameters
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackResult,
    AttackSettings,
)
from attack_engine.scheduling.models import AttackFamily


def apply_spike(
    request: AttackRequest,
    params: SpikeParameters,
    rng: np.random.Generator,
    settings: AttackSettings,
) -> AttackResult:
    clean = base.clean_array(request)
    if not np.isfinite(clean).all():
        return base.rejected_result(
            request,
            AttackFamily.SPIKE,
            AttackRejectionReason.MISSING_TARGET,
            "window has NaN target",
        )

    sign = base.resolve_sign(params.sign, rng)
    point_count = min(int(rng.integers(params.min_points, params.max_points + 1)), clean.size)
    positions = np.sort(rng.choice(clean.size, size=point_count, replace=False))

    candidate = clean.copy()
    candidate[positions] = candidate[positions] + sign * params.magnitude
    final, changed, saturated = base.finalize_candidate(clean, candidate, settings)

    if not changed.any():
        return base.rejected_result(
            request,
            AttackFamily.SPIKE,
            AttackRejectionReason.ZERO_EFFECT,
            "all spiked positions equal their clean value after bounds/quantization",
            requested_magnitude=params.magnitude,
            sign=sign,
        )
    return base.build_result(
        request,
        AttackFamily.SPIKE,
        clean,
        final,
        changed,
        saturated,
        requested_magnitude=params.magnitude,
        sign=sign,
        extra={
            "requested_points": point_count,
            "positions": ",".join(map(str, positions.tolist())),
        },
    )
