"""Shared numerical helpers for attack transformations.

The ``changed`` mask is always derived the same way: a position counts as modified
only when both the clean value and the final value are finite and the final value
differs from the clean value, after optional quantization and bounds clipping. This
is the single definition every family uses.
"""

from __future__ import annotations

import numpy as np

from attack_engine.attacks.models import (
    AttackDiagnostics,
    AttackRejectionReason,
    AttackRequest,
    AttackResult,
    AttackSettings,
    AttackStatus,
    ReplaySource,
    SignPolicy,
    TargetBounds,
)
from attack_engine.scheduling.models import AttackFamily


def resolve_sign(policy: SignPolicy, rng: np.random.Generator) -> int:
    """Return +1 or -1, drawing deterministically from ``rng`` for the random policy."""
    if policy is SignPolicy.POSITIVE:
        return 1
    if policy is SignPolicy.NEGATIVE:
        return -1
    return 1 if rng.integers(2) == 0 else -1


def quantize(values: np.ndarray, quantum: float | None) -> np.ndarray:
    """Snap values to a multiple of ``quantum`` (round half to even), or pass through."""
    if quantum is None:
        return values
    if not np.isfinite(quantum) or quantum <= 0.0:
        raise ValueError(f"quantum must be positive and finite, got {quantum}")
    return np.round(values / quantum) * quantum


def clip_to_bounds(values: np.ndarray, bounds: TargetBounds) -> tuple[np.ndarray, np.ndarray]:
    """Clip to ``[low, high]``; return the clipped array and the saturation mask."""
    clipped = np.clip(values, bounds.low, bounds.high)
    saturated = clipped != values
    return clipped, saturated


def finalize_candidate(
    clean: np.ndarray, candidate: np.ndarray, settings: AttackSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize then clip the candidate and compute the changed and saturation masks.

    Precision policy: an exactly zero intended perturbation always preserves the clean
    float, including off-grid observations. With quantization enabled, candidates in the
    same round-half-to-even bin as the clean value also preserve that exact clean float;
    quantizing the clean array is only a comparison, never a rewrite. A bin crossing is
    retained even when the requested perturbation is smaller than a quantum.

    After clipping, changed-mask comparisons are exact, with no absolute tolerance. With
    quantization disabled every representable finite difference counts, including one ULP.
    Thus quantize(0.141, 0.001) cannot manufacture a modification or hide a small attack
    behind an unrelated fixed epsilon. The final array and the mask always agree.
    """
    candidate = candidate.astype("float64", copy=True)
    intended_change = candidate != clean
    final = quantize(candidate, settings.quantum)
    if settings.quantum is not None:
        intended_change &= final != quantize(clean, settings.quantum)
    final, saturated = clip_to_bounds(final, settings.bounds)
    both_finite = np.isfinite(clean) & np.isfinite(final)
    changed = intended_change & both_finite & (final != clean)
    # Unchanged positions keep exactly the clean value so no spurious delta remains.
    final = np.where(changed, final, clean)
    return final, changed, saturated & changed


def _delta_stats(clean: np.ndarray, final: np.ndarray, changed: np.ndarray) -> tuple[float, float]:
    if not changed.any():
        return 0.0, 0.0
    deltas = np.abs(final[changed] - clean[changed])
    return float(deltas.mean()), float(deltas.max())


def build_result(
    request: AttackRequest,
    family: AttackFamily,
    clean: np.ndarray,
    final: np.ndarray,
    changed: np.ndarray,
    saturated: np.ndarray,
    *,
    requested_magnitude: float,
    sign: int,
    extra: dict[str, float | int | str] | None = None,
    replay_source: ReplaySource | None = None,
) -> AttackResult:
    """Assemble an APPLIED :class:`AttackResult` from finalized arrays."""
    mean_abs, max_abs = _delta_stats(clean, final, changed)
    diagnostics = AttackDiagnostics(
        requested_magnitude=requested_magnitude,
        sign=sign,
        modified_count=int(changed.sum()),
        mean_abs_delta=mean_abs,
        max_abs_delta=max_abs,
        saturated_count=int(saturated.sum()),
        extra=extra or {},
    )
    return AttackResult(
        window_id=request.window.window_id,
        family=family,
        station=request.window.station,
        partition=request.window.partition,
        status=AttackStatus.APPLIED,
        row_labels=request.window.row_labels,
        clean_values=tuple(float(value) for value in clean),
        candidate_values=tuple(float(value) for value in final),
        changed_mask=tuple(bool(flag) for flag in changed),
        diagnostics=diagnostics,
        replay_source=replay_source,
        target_columns=request.resolved_target_columns,
    )


def rejected_result(
    request: AttackRequest,
    family: AttackFamily,
    reason: AttackRejectionReason,
    detail: str,
    requested_magnitude: float = 0.0,
    sign: int = 0,
    extra: dict[str, float | int | str] | None = None,
) -> AttackResult:
    """Assemble a REJECTED result that leaves every value unchanged."""
    clean = request.clean_values
    return AttackResult(
        window_id=request.window.window_id,
        family=family,
        station=request.window.station,
        partition=request.window.partition,
        status=AttackStatus.REJECTED,
        row_labels=request.window.row_labels,
        clean_values=clean,
        candidate_values=clean,
        changed_mask=tuple(False for _ in clean),
        diagnostics=AttackDiagnostics(
            requested_magnitude=requested_magnitude,
            sign=sign,
            modified_count=0,
            mean_abs_delta=0.0,
            max_abs_delta=0.0,
            saturated_count=0,
            extra=extra or {},
        ),
        rejection_reason=reason,
        detail=detail,
        target_columns=request.resolved_target_columns,
    )


def clean_array(request: AttackRequest) -> np.ndarray:
    return np.asarray(request.clean_values, dtype="float64")
