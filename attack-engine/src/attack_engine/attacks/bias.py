"""Bias injection: a constant additive offset across the whole window.

The realized offset must stay constant, so instead of clipping (which would bend
the offset) the event is checked for feasibility first: if ``clean + offset`` would
leave the target bounds at any position, the candidate is rejected with a reason.
"""

from __future__ import annotations

import numpy as np

from attack_engine.attacks import base
from attack_engine.attacks.config import BiasParameters
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackResult,
    AttackSettings,
)
from attack_engine.scheduling.models import AttackFamily


def apply_bias(
    request: AttackRequest,
    params: BiasParameters,
    rng: np.random.Generator,
    settings: AttackSettings,
) -> AttackResult:
    clean = base.clean_array(request)
    if not np.isfinite(clean).all():
        return base.rejected_result(
            request,
            AttackFamily.BIAS,
            AttackRejectionReason.MISSING_TARGET,
            "window has NaN target",
        )

    sign = base.resolve_sign(params.sign, rng)
    offset = sign * params.magnitude
    candidate = clean + offset

    bounds = settings.bounds
    # Allow float64 arithmetic roundoff, not a fixed scientific detection threshold.
    # A fixed 1e-9 allowance could silently clip genuine small biases near a bound.
    scale = max(float(np.abs(clean).max()), abs(offset), abs(bounds.low), abs(bounds.high))
    roundoff = 4 * np.finfo(np.float64).eps * scale
    if candidate.min() < bounds.low - roundoff or candidate.max() > bounds.high + roundoff:
        return base.rejected_result(
            request,
            AttackFamily.BIAS,
            AttackRejectionReason.INFEASIBLE_BOUNDS,
            f"constant offset {offset:+.4f} leaves [{bounds.low}, {bounds.high}] "
            f"(clean range [{clean.min():.4f}, {clean.max():.4f}])",
            requested_magnitude=params.magnitude,
            sign=sign,
        )

    final, changed, saturated = base.finalize_candidate(clean, candidate, settings)
    if not changed.any():
        return base.rejected_result(
            request,
            AttackFamily.BIAS,
            AttackRejectionReason.ZERO_EFFECT,
            "offset vanished after quantization",
            requested_magnitude=params.magnitude,
            sign=sign,
        )

    # Include unchanged positions: changing just part of a window is not a constant bias.
    realized = final - clean
    if float(realized.max() - realized.min()) > roundoff:
        return base.rejected_result(
            request,
            AttackFamily.BIAS,
            AttackRejectionReason.INFEASIBLE_BOUNDS,
            "quantization made the realized offset non-constant; use a bias aligned to the quantum",
            requested_magnitude=params.magnitude,
            sign=sign,
        )
    return base.build_result(
        request,
        AttackFamily.BIAS,
        clean,
        final,
        changed,
        saturated,
        requested_magnitude=params.magnitude,
        sign=sign,
        extra={"realized_offset": float(realized[0])},
    )
