"""Spike injection: abrupt change at 1-3 synchronized positions.

``k`` positions (drawn in ``[min_points, max_points]`` and capped at the window length) are
chosen once without replacement and shifted. With extra ``target_columns`` the same positions
and sign are used on every selected variable, each with its own unit-aware magnitude, so the
spikes are synchronized across variables. Values are clipped to each variable's bounds.
"""

from __future__ import annotations

import numpy as np

from attack_engine.attacks import base, multitarget
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

    column_candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    for column in params.target_columns:
        column_clean = (
            clean
            if column == params.target_columns[0]
            else (multitarget.secondary_clean_array(request, column))
        )
        magnitude = (
            params.magnitude
            if column == params.target_columns[0]
            else (params.secondary_magnitudes[column])
        )
        candidate = column_clean.copy()
        candidate[positions] = candidate[positions] + sign * magnitude
        column_candidates.append((column, column_clean, candidate))

    return multitarget.build_multi_result(
        request,
        AttackFamily.SPIKE,
        column_candidates,
        settings,
        requested_magnitude=params.magnitude,
        sign=sign,
        extra={
            "requested_points": point_count,
            "positions": ",".join(map(str, positions.tolist())),
        },
    )
