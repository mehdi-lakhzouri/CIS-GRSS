"""Slow drift: a progressive deviation that grows with elapsed time.

The ramp is linear in real elapsed time, reaching ``sign * final_magnitude`` at the
last observation: ``ramp(t) = A * (t - t0) / (t_last - t0)``. With ``hold_first`` the
first observation keeps a zero ramp and stays unchanged. The event is rejected if the
window is too short for a meaningful ramp or if the required slope exceeds the limit.
"""

from __future__ import annotations

import numpy as np

from attack_engine.attacks import base
from attack_engine.attacks.config import DriftParameters
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackResult,
    AttackSettings,
)
from attack_engine.scheduling.models import AttackFamily


def apply_drift(
    request: AttackRequest,
    params: DriftParameters,
    rng: np.random.Generator,
    settings: AttackSettings,
) -> AttackResult:
    clean = base.clean_array(request)
    elapsed = np.asarray(request.elapsed_hours, dtype="float64")
    if not np.isfinite(clean).all():
        return base.rejected_result(
            request,
            AttackFamily.DRIFT,
            AttackRejectionReason.MISSING_TARGET,
            "window has NaN target",
        )

    total_elapsed = float(elapsed[-1]) if elapsed.size else 0.0
    if clean.size < 2 or total_elapsed <= 0.0:
        return base.rejected_result(
            request,
            AttackFamily.DRIFT,
            AttackRejectionReason.WINDOW_TOO_SHORT,
            "drift needs at least two distinct timestamps",
            requested_magnitude=params.final_magnitude,
        )

    slope = params.final_magnitude / total_elapsed
    if params.max_slope_per_hour is not None and slope > params.max_slope_per_hour:
        return base.rejected_result(
            request,
            AttackFamily.DRIFT,
            AttackRejectionReason.SLOPE_TOO_STEEP,
            f"slope {slope:.5f}/h exceeds maximum {params.max_slope_per_hour:.5f}/h",
            requested_magnitude=params.final_magnitude,
        )

    sign = base.resolve_sign(params.sign, rng)
    ramp = sign * params.final_magnitude * (elapsed / total_elapsed)
    if params.hold_first:
        ramp[0] = 0.0
    candidate = clean + ramp
    final, changed, saturated = base.finalize_candidate(clean, candidate, settings)

    if not changed.any():
        return base.rejected_result(
            request,
            AttackFamily.DRIFT,
            AttackRejectionReason.ZERO_EFFECT,
            "ramp vanished after bounds/quantization",
            requested_magnitude=params.final_magnitude,
            sign=sign,
        )
    return base.build_result(
        request,
        AttackFamily.DRIFT,
        clean,
        final,
        changed,
        saturated,
        requested_magnitude=params.final_magnitude,
        sign=sign,
        extra={"slope_per_hour": slope, "hold_first": int(params.hold_first)},
    )
