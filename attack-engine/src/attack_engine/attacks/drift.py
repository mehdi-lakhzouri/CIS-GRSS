"""Slow drift: a progressive deviation that grows with elapsed time.

The ramp is linear in real elapsed time, reaching ``sign * final_magnitude`` at the last
observation. With ``hold_first`` the first observation keeps a zero ramp and stays unchanged.
Extra ``target_columns`` share the timing and sign but ramp to their own final magnitude, so
the drifts are temporally aligned across variables. The event is rejected if the window is too
short or if any variable's required slope exceeds the configured maximum.
"""

from __future__ import annotations

import numpy as np

from attack_engine.attacks import base, multitarget
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

    primary = params.target_columns[0]
    magnitudes = {primary: params.final_magnitude, **params.secondary_magnitudes}
    for column, limit in params.secondary_max_slopes_per_hour.items():
        if magnitudes[column] / total_elapsed > limit:
            return base.rejected_result(
                request,
                AttackFamily.DRIFT,
                AttackRejectionReason.SLOPE_TOO_STEEP,
                f"{column} slope exceeds {limit}/h",
                requested_magnitude=params.final_magnitude,
            )
    # Limits are expressed separately in each channel's units per hour.
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
    shape = elapsed / total_elapsed
    if params.hold_first:
        shape = shape.copy()
        shape[0] = 0.0

    column_candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    for column in params.target_columns:
        column_clean = (
            clean if column == primary else multitarget.secondary_clean_array(request, column)
        )
        candidate = column_clean + sign * magnitudes[column] * shape
        column_candidates.append((column, column_clean, candidate))

    result = multitarget.build_multi_result(
        request,
        AttackFamily.DRIFT,
        column_candidates,
        settings,
        requested_magnitude=params.final_magnitude,
        sign=sign,
        extra={
            "slope_per_hour": params.final_magnitude / total_elapsed,
            "hold_first": int(params.hold_first),
        },
    )
    # Legacy single-target ramps retain their historical saturation/precision behavior.
    # Coordinated ramps must also satisfy each slope limit after finalization.
    if result.applied and request.resolved_target_columns != ("SOIL_MOISTURE_5",):
        limits = {primary: params.max_slope_per_hour, **params.secondary_max_slopes_per_hour}
        for column in result.columns:
            limit = limits.get(column.column)
            if limit is None:
                continue
            offsets = np.asarray(column.candidate_values) - column.clean_values
            rates = np.abs(np.diff(offsets)) / np.diff(elapsed)
            if np.any(np.isfinite(rates) & (rates > limit + 1e-12)):
                return base.rejected_result(
                    request,
                    AttackFamily.DRIFT,
                    AttackRejectionReason.SLOPE_TOO_STEEP,
                    f"{column.column} finalized slope exceeds {limit}/h",
                    requested_magnitude=params.final_magnitude,
                    sign=sign,
                )
    return result
