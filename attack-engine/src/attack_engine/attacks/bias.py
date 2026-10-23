"""Bias injection: a constant additive offset across the whole window.

The realized offset must stay constant, so instead of clipping (which would bend the offset)
each variable is checked for feasibility first: if ``clean + offset`` would leave that
variable's bounds at any position, the whole coordinated event is rejected. Extra
``target_columns`` share the sign but each applies its own unit-aware constant offset.
"""

from __future__ import annotations

import numpy as np

from attack_engine.attacks import base, multitarget
from attack_engine.attacks.config import BiasParameters
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackResult,
    AttackSettings,
    TargetBounds,
)
from attack_engine.scheduling.models import AttackFamily


def _roundoff(clean: np.ndarray, offset: float, bounds: TargetBounds) -> float:
    finite = clean[np.isfinite(clean)]
    scale = max(
        float(np.abs(finite).max()) if finite.size else 0.0,
        abs(offset),
        abs(bounds.low),
        abs(bounds.high),
    )
    return 4 * np.finfo(np.float64).eps * scale


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
    primary = params.target_columns[0]
    magnitudes = {primary: params.magnitude, **params.secondary_magnitudes}

    column_candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    primary_offset = 0.0
    for column in params.target_columns:
        column_clean = (
            clean if column == primary else multitarget.secondary_clean_array(request, column)
        )
        offset = sign * magnitudes[column]
        candidate = column_clean + offset
        bounds = multitarget.settings_for(column, settings).bounds
        low, high = bounds.low, bounds.high
        roundoff = _roundoff(column_clean, offset, TargetBounds(low, high))
        finite = candidate[np.isfinite(candidate)]
        if finite.size and (finite.min() < low - roundoff or finite.max() > high + roundoff):
            return base.rejected_result(
                request,
                AttackFamily.BIAS,
                AttackRejectionReason.INFEASIBLE_BOUNDS,
                f"{column} constant offset {offset:+.4f} leaves [{low}, {high}]",
                requested_magnitude=params.magnitude,
                sign=sign,
            )
        if column == primary:
            primary_offset = offset
        column_candidates.append((column, column_clean, candidate))

    result = multitarget.build_multi_result(
        request,
        AttackFamily.BIAS,
        column_candidates,
        settings,
        requested_magnitude=params.magnitude,
        sign=sign,
        extra={"realized_offset": primary_offset},
    )
    return _reject_if_non_constant(request, result, sign, params.magnitude)


def _reject_if_non_constant(
    request: AttackRequest, result: AttackResult, sign: int, magnitude: float
) -> AttackResult:
    """A partial, quantization-bent offset is not a constant bias: reject the whole event."""
    if not result.applied:
        return result
    for column in result.columns:
        clean = np.asarray(column.clean_values, dtype="float64")
        final = np.asarray(column.candidate_values, dtype="float64")
        changed = np.asarray(column.changed_mask, dtype=bool)
        if not changed.any():
            continue
        realized = (final - clean)[np.isfinite(clean)]
        low, high = multitarget.VARIABLE_BOUNDS[column.column]
        roundoff = _roundoff(clean, sign * magnitude, TargetBounds(low, high))
        if float(realized.max() - realized.min()) > roundoff:
            return base.rejected_result(
                request,
                AttackFamily.BIAS,
                AttackRejectionReason.INFEASIBLE_BOUNDS,
                f"{column.column} offset became non-constant after quantization",
                requested_magnitude=magnitude,
                sign=sign,
            )
    return result
