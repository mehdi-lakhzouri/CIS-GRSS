"""Shared helpers for coordinated multi-variable attacks.

Every family produces per-variable candidate arrays; this module finalizes each variable
against its own physical bounds (unit-aware) and assembles one :class:`AttackResult` with a
primary :class:`ColumnModification` plus one per modified secondary variable. The primary
selected target must change at least one finite cell, or the entire event is rejected.
The executor accounts for every variable under the configured row/cell policy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from attack_engine.attacks import base
from attack_engine.attacks.models import (
    AttackDiagnostics,
    AttackRejectionReason,
    AttackRequest,
    AttackResult,
    AttackSettings,
    AttackStatus,
    ColumnModification,
    ReplaySource,
    TargetBounds,
)
from attack_engine.scheduling.models import AttackFamily

# Per-variable physical bounds (a project contract, not a universal property). Different units:
# soil moisture is m3/m3, precipitation is mm, so numeric offsets are never shared across them.
VARIABLE_BOUNDS: dict[str, tuple[float, float]] = {
    "SOIL_MOISTURE_5": (0.0, 0.70),
    "SOIL_MOISTURE_10": (0.0, 0.70),
    "P_CALC": (0.0, 200.0),
}


def settings_for(column: str, base_settings: AttackSettings) -> AttackSettings:
    """Resolve bounds and precision in the selected variable's own units."""
    low, high = VARIABLE_BOUNDS[column]
    if column == "SOIL_MOISTURE_5":
        low, high = base_settings.bounds.low, base_settings.bounds.high
    quantum = base_settings.quantum
    limits = base_settings.variable_constraints.get(column)
    if limits is not None:
        low = limits.low if limits.low is not None else low
        high = limits.high if limits.high is not None else high
        if "quantum" in limits.model_fields_set:
            quantum = limits.quantum
    return AttackSettings(bounds=TargetBounds(low, high), quantum=quantum)


def secondary_clean_array(request: AttackRequest, column: str) -> np.ndarray:
    values = request.extra_clean.get(column)
    if values is None:
        return np.full(request.length, np.nan)
    return np.asarray(values, dtype="float64")


def build_multi_result(
    request: AttackRequest,
    family: AttackFamily,
    column_candidates: list[tuple[str, np.ndarray, np.ndarray]],
    settings: AttackSettings,
    *,
    requested_magnitude: float,
    sign: int,
    extra: dict[str, float | int | str] | None = None,
    replay_source: ReplaySource | None = None,
) -> AttackResult:
    """Assemble a coordinated result; ``column_candidates`` is primary-first per variable."""
    if request.resolved_target_columns != ("SOIL_MOISTURE_5",):
        for column, clean, candidate in column_candidates:
            bounds = settings_for(column, settings).bounds
            finite = np.isfinite(clean) & np.isfinite(candidate)
            if np.any(
                finite & ((candidate < bounds.low - 1e-12) | (candidate > bounds.high + 1e-12))
            ):
                return base.rejected_result(
                    request,
                    family,
                    AttackRejectionReason.INFEASIBLE_BOUNDS,
                    f"{column} candidate leaves physical bounds; atomic rejection",
                    requested_magnitude=requested_magnitude,
                    sign=sign,
                    extra=extra,
                )
    primary_column, primary_clean, primary_candidate = column_candidates[0]

    def numerical_settings(column: str) -> AttackSettings:
        resolved = settings_for(column, settings)
        # A coordinated replay copies genuine source values exactly. The old single-target
        # precision policy is retained for the seed-14 reference campaign.
        if family is AttackFamily.REPLAY and request.resolved_target_columns != (
            "SOIL_MOISTURE_5",
        ):
            resolved = replace(resolved, quantum=None)
        return resolved

    primary_final, primary_changed, primary_saturated = base.finalize_candidate(
        primary_clean, primary_candidate, numerical_settings(primary_column)
    )
    if not primary_changed.any():
        return base.rejected_result(
            request,
            family,
            AttackRejectionReason.ZERO_EFFECT,
            "primary target unchanged after bounds and quantization",
            requested_magnitude=requested_magnitude,
            sign=sign,
            extra=extra,
        )

    extra_columns: list[ColumnModification] = []
    saturated_total = int(primary_saturated.sum())
    for column, clean, candidate in column_candidates[1:]:
        final, changed, saturated = base.finalize_candidate(
            clean, candidate, numerical_settings(column)
        )
        saturated_total += int(saturated.sum())
        if not changed.any():
            return base.rejected_result(
                request,
                family,
                AttackRejectionReason.ZERO_EFFECT,
                f"selected variable {column} has no feasible modification; atomic rejection",
                requested_magnitude=requested_magnitude,
                sign=sign,
                extra=extra,
            )
        extra_columns.append(
            ColumnModification(
                column=column,
                row_labels=request.window.row_labels,
                clean_values=tuple(float(v) for v in clean),
                candidate_values=tuple(float(v) for v in final),
                changed_mask=tuple(bool(f) for f in changed),
            )
        )

    mean_abs, max_abs = base._delta_stats(primary_clean, primary_final, primary_changed)
    secondary_cells = sum(column.modified_count for column in extra_columns)
    diagnostics = AttackDiagnostics(
        requested_magnitude=requested_magnitude,
        sign=sign,
        modified_count=int(primary_changed.sum()),
        mean_abs_delta=mean_abs,
        max_abs_delta=max_abs,
        saturated_count=saturated_total,
        extra={
            **(extra or {}),
            "target_columns": ",".join(column for column, _, _ in column_candidates),
            "secondary_cells": secondary_cells,
            **{f"cells_{column.column}": column.modified_count for column in extra_columns},
        },
    )
    result = AttackResult(
        window_id=request.window.window_id,
        family=family,
        station=request.window.station,
        partition=request.window.partition,
        status=AttackStatus.APPLIED,
        row_labels=request.window.row_labels,
        clean_values=tuple(float(v) for v in primary_clean),
        candidate_values=tuple(float(v) for v in primary_final),
        changed_mask=tuple(bool(f) for f in primary_changed),
        diagnostics=diagnostics,
        replay_source=replay_source,
        target_columns=tuple(column for column, _, _ in column_candidates),
        extra_columns=tuple(extra_columns),
    )
    return validate_result(request, result, settings)


def validate_result(
    request: AttackRequest, result: AttackResult, settings: AttackSettings
) -> AttackResult:
    """Validate finalized values before a single cell or budget unit is committed.

    Temporal limits cover adjacent finite emissions (including both window boundaries),
    only where the attack touches an edge. Missing observations break continuity. Cross
    relationships are checked at affected rows with both operands finite; they never fill gaps.
    """
    if not result.applied:
        return result

    def reject(reason: AttackRejectionReason, detail: str) -> AttackResult:
        return base.rejected_result(
            request,
            result.family,
            reason,
            detail,
            result.diagnostics.requested_magnitude,
            result.diagnostics.sign,
            result.diagnostics.extra,
        )

    columns = {column.column: column for column in result.columns}
    for name in request.resolved_target_columns:
        column = columns.get(name)
        if column is None or not column.modified_count:
            return reject(AttackRejectionReason.ZERO_EFFECT, f"{name} unchanged; atomic rejection")
        clean = np.asarray(column.clean_values)
        final = np.asarray(column.candidate_values)
        changed = np.asarray(column.changed_mask)
        bounds = settings_for(name, settings).bounds
        if np.any(changed & ((final < bounds.low) | (final > bounds.high))):
            return reject(
                AttackRejectionReason.INFEASIBLE_BOUNDS, f"{name} outside physical bounds"
            )
        limits = settings.variable_constraints.get(name)
        if limits is None:
            continue
        elapsed = np.asarray(request.elapsed_hours)
        intervals = np.diff(
            np.r_[
                -request.boundary_interval_hours,
                elapsed,
                elapsed[-1] + request.boundary_interval_hours,
            ]
        )
        if np.any(intervals <= 0):
            return reject(AttackRejectionReason.INVALID_CONTEXT, "non-increasing elapsed time")
        values = np.r_[
            request.previous_values.get(name, np.nan), final, request.next_values.get(name, np.nan)
        ]
        offsets = np.r_[
            request.previous_offsets.get(name, 0.0),
            final - clean,
            request.next_offsets.get(name, 0.0),
        ]
        touched = np.r_[False, changed, False]
        edges = (touched[:-1] | touched[1:]) & np.isfinite(values[:-1]) & np.isfinite(values[1:])
        for series, maximum, label in (
            (values, limits.max_step_per_hour, "measurement step"),
            (offsets, limits.max_offset_rate_per_hour, "offset rate"),
        ):
            if maximum is not None and np.any(
                edges & (np.abs(np.diff(series)) / intervals > maximum + 1e-12)
            ):
                return reject(AttackRejectionReason.TEMPORAL_CONSTRAINT, f"{name} {label} exceeded")

    for relation in settings.relationships:
        arrays = []
        masks = []
        for name in (relation.left, relation.right):
            column = columns.get(name)
            clean = np.asarray(request.reference_clean.get(name, (np.nan,) * request.length))
            final = np.asarray(column.candidate_values) if column else clean
            if column:
                clean = np.asarray(column.clean_values)
            arrays.append(final if relation.quantity == "values" else final - clean)
            masks.append(
                np.asarray(column.changed_mask) if column else np.zeros(request.length, bool)
            )
        left, right = arrays
        difference = left - relation.scale * right
        checked = (masks[0] | masks[1]) & np.isfinite(left) & np.isfinite(right)
        if np.any(
            checked
            & ((difference < relation.lower - 1e-12) | (difference > relation.upper + 1e-12))
        ):
            return reject(
                AttackRejectionReason.CROSS_VARIABLE_CONSTRAINT,
                f"{relation.left} - {relation.scale} * {relation.right} outside "
                f"[{relation.lower}, {relation.upper}] ({relation.quantity})",
            )
    return result
