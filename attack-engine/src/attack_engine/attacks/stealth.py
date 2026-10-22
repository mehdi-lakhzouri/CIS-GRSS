"""Constrained local FDI simulation using a frozen, train-only station reference.

Each attempt emits sequentially, feeding its finalized values back into causal lags.
All bounds are checked after the shared quantization policy. Failure rejects the whole
attempt; no constraint is relaxed to consume a quota. No anomaly detector is consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from attack_engine.attacks import base
from attack_engine.attacks.calibration import StationReference, feature_vector
from attack_engine.attacks.config import MissingContextPolicy, StealthParameters
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackResult,
    AttackSettings,
    StealthContext,
)
from attack_engine.scheduling.models import AttackFamily


@dataclass(frozen=True)
class _Attempt:
    values: np.ndarray
    reason: AttackRejectionReason | None
    detail: str
    cumulative_drift: float = 0.0
    max_offset_rate: float = 0.0
    max_step_rate: float = 0.0
    max_context_residual: float = 0.0


def _project(
    clean: float, desired: float, low: float, high: float, settings: AttackSettings
) -> float | None:
    if settings.quantum is not None:
        q = settings.quantum
        first, last = np.ceil(low / q), np.floor(high / q)
        if first > last:
            return None
        desired = float(np.clip(np.round(desired / q), first, last) * q)
    else:
        desired = float(np.clip(desired, low, high))
    final, _, _ = base.finalize_candidate(np.array([clean]), np.array([desired]), settings)
    value = float(final[0])
    # Arithmetic allowance only, never a changed-mask or minimum-impact tolerance.
    roundoff = 8 * np.finfo(float).eps * max(1.0, abs(low), abs(high))
    if not (
        settings.bounds.low <= value <= settings.bounds.high
        and low - roundoff <= value <= high + roundoff
    ):
        return None
    return value


def _rollout(
    clean: np.ndarray,
    context: StealthContext,
    model: StationReference,
    available: np.ndarray,
    params: StealthParameters,
    settings: AttackSettings,
    magnitude: float,
    sign: int,
) -> _Attempt:
    final = clean.copy()
    history = list(context.history_values[-context.calibration.history_samples :])
    previous_time = context.history_times[-1]
    previous_value, previous_offset = history[-1], 0.0
    cumulative = max_offset_rate = max_step_rate = max_residual = 0.0
    tolerance = model.residual_tolerance * params.residual_tolerance_multiplier
    roundoff = 8 * np.finfo(float).eps * max(1.0, abs(settings.bounds.high))
    for index, time in enumerate(context.times):
        dt = (time - previous_time).total_seconds() / 3600
        if not available[index]:
            # Explicitly leave this observation genuine. Its return to zero offset still
            # consumes total-variation budget, and its emitted clean value enters history.
            value = float(clean[index])
        else:
            measurements = tuple(
                context.measurements[name][index] for name in params.required_context_features
            )
            mu = model.predict(feature_vector(measurements, tuple(history), time))
            remaining = max(0.0, params.max_cumulative_drift - cumulative)
            offset_step = min(params.max_change_rate * dt, remaining)
            low = max(
                settings.bounds.low,
                mu - tolerance,
                previous_value - params.max_step_change * dt,
                clean[index] + previous_offset - offset_step,
            )
            high = min(
                settings.bounds.high,
                mu + tolerance,
                previous_value + params.max_step_change * dt,
                clean[index] + previous_offset + offset_step,
            )
            # The sign is an attacker capability/parameter, not a suggested direction.
            if sign > 0:
                low = max(low, clean[index])
            else:
                high = min(high, clean[index])
            if not np.isfinite(mu) or low > high:
                return _Attempt(
                    final,
                    AttackRejectionReason.INFEASIBLE_CONTEXT,
                    f"empty feasible range at position {index}",
                )
            value = _project(
                float(clean[index]), float(clean[index] + sign * magnitude), low, high, settings
            )
            if value is None:
                return _Attempt(
                    final,
                    AttackRejectionReason.QUANTIZATION_INFEASIBLE,
                    f"no representable feasible output at position {index}",
                )
            max_residual = max(max_residual, abs(value - mu))
            max_step_rate = max(max_step_rate, abs(value - previous_value) / dt)
        offset = value - clean[index]
        offset_rate = abs(offset - previous_offset) / dt
        cumulative += abs(offset - previous_offset)
        if (
            offset_rate > params.max_change_rate + roundoff
            or cumulative > params.max_cumulative_drift + roundoff
        ):
            return _Attempt(
                final,
                AttackRejectionReason.INFEASIBLE_CONTEXT,
                f"offset rate or cumulative drift exceeded at position {index}",
            )
        final[index] = value
        max_offset_rate = max(max_offset_rate, offset_rate)
        history = [*history[1:], value]
        previous_value, previous_offset, previous_time = value, offset, time
    return _Attempt(final, None, "", cumulative, max_offset_rate, max_step_rate, max_residual)


def apply_stealth(
    request: AttackRequest,
    params: StealthParameters,
    rng: np.random.Generator,
    settings: AttackSettings,
) -> AttackResult:
    clean = base.clean_array(request)
    context = request.stealth_context
    extra: dict[str, float | int | str] = {
        "context_total": request.length,
        "context_available": 0,
        "attempts": 0,
    }

    def reject(reason: AttackRejectionReason, detail: str) -> AttackResult:
        return base.rejected_result(request, AttackFamily.STEALTH, reason, detail, extra=extra)

    if not np.isfinite(clean).all():
        return reject(AttackRejectionReason.MISSING_TARGET, "window has a missing target")
    if context is None:
        return reject(AttackRejectionReason.NO_CALIBRATION, "no contextual request supplied")
    calibration = context.calibration
    extra["calibration_sha256"] = calibration.calibration_sha256
    if (
        len(context.times) != request.length
        or not context.times
        or any(
            len(context.measurements.get(name, ())) != request.length
            for name in params.required_context_features
        )
    ):
        return reject(
            AttackRejectionReason.INVALID_CONTEXT, "context dimensions do not match window"
        )
    available = np.isfinite(
        np.column_stack([context.measurements[name] for name in params.required_context_features])
    ).all(axis=1)
    extra["context_available"] = int(available.sum())
    if (
        request.window.partition not in params.attack_partitions
        or context.times[0] <= calibration.partition_end
    ):
        return reject(
            AttackRejectionReason.FORBIDDEN_PARTITION,
            "stealth is restricted to validation/test strictly after train calibration",
        )
    model = calibration.station_models.get(request.window.station)
    if model is None:
        return reject(
            AttackRejectionReason.NO_CALIBRATION,
            "no calibrated station model; no implicit pooled fallback",
        )
    n = calibration.history_samples
    if len(context.history_values) < n or len(context.history_times) != len(context.history_values):
        return reject(AttackRejectionReason.INSUFFICIENT_HISTORY, f"need {n} earlier emissions")
    history_times = context.history_times[-n:]
    history_steps = [
        (b - a).total_seconds() / 3600
        for a, b in zip(history_times, (*history_times[1:], context.times[0]), strict=True)
    ]
    elapsed = tuple((time - context.times[0]).total_seconds() / 3600 for time in context.times)
    if not np.isfinite(context.history_values[-n:]).all() or any(
        step != calibration.interval_hours for step in history_steps
    ):
        return reject(
            AttackRejectionReason.INSUFFICIENT_HISTORY, "history is not finite and contiguous"
        )
    if (
        context.times[0] != request.window.first_observation_time
        or context.times[-1] != request.window.last_observation_time
        or elapsed != request.elapsed_hours
        or any(b <= a for a, b in pairwise(context.times))
    ):
        return reject(AttackRejectionReason.INVALID_CONTEXT, "timestamps/elapsed time do not match")
    if not available.all() and params.missing_context_policy is MissingContextPolicy.REJECT_WINDOW:
        return reject(
            AttackRejectionReason.MISSING_CONTEXT,
            "required environmental measurements are missing; rain is never filled with 0",
        )
    last_reason = AttackRejectionReason.ZERO_EFFECT
    detail = "no effective perturbation"
    for attempt_number in range(1, params.max_candidate_attempts + 1):
        extra["attempts"] = attempt_number
        magnitude = float(rng.uniform(params.magnitude_min, params.magnitude_max))
        sign = base.resolve_sign(params.sign, rng)
        attempt = _rollout(clean, context, model, available, params, settings, magnitude, sign)
        if attempt.reason is not None:
            last_reason, detail = attempt.reason, attempt.detail
            continue
        final, changed, saturated = base.finalize_candidate(clean, attempt.values, settings)
        impact = float(np.mean(np.abs(final - clean)))
        if not changed.any() or impact < params.min_attack_impact:
            last_reason = AttackRejectionReason.BELOW_MIN_IMPACT
            detail = f"mean absolute impact {impact:.8g} below {params.min_attack_impact:.8g}"
            continue
        extra.update(
            cumulative_drift=attempt.cumulative_drift,
            max_offset_rate=attempt.max_offset_rate,
            max_step_rate=attempt.max_step_rate,
            max_context_residual=attempt.max_context_residual,
            mean_abs_impact=impact,
        )
        return base.build_result(
            request,
            AttackFamily.STEALTH,
            clean,
            final,
            changed,
            saturated,
            requested_magnitude=magnitude,
            sign=sign,
            extra=extra,
        )
    return reject(last_reason, f"{detail}; exhausted {params.max_candidate_attempts} attempts")
