"""Historical replay: overwrite the window with older genuine measurements.

The source is an equal-length, hourly-contiguous, fully-finite run of the same
station and partition, read from the immutable clean dataset, ending strictly
before the destination window and at least ``min_age_hours`` older. Source
selection is deterministic. Timestamps and context columns are never touched; this
is dataset-level replay, not a replay of signed network packets.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from attack_engine.attacks import base
from attack_engine.attacks.config import ReplayParameters
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackResult,
    AttackSettings,
    ReplaySource,
)
from attack_engine.scheduling.models import AttackFamily

_Candidate = tuple[int, int, datetime]  # (run index, start offset, source first time)


def _age_hours(dest_first: datetime, source_first: datetime) -> float:
    return (dest_first - source_first).total_seconds() / 3600.0


def _source_candidates(request: AttackRequest, params: ReplayParameters) -> list[_Candidate]:
    history = request.history
    length = request.length
    dest_first = request.window.first_observation_time
    candidates: list[_Candidate] = []
    if history is None:
        return candidates
    for run_index, times in enumerate(history.run_times):
        for start in range(len(times) - length + 1):
            source_first = times[start]
            source_last = times[start + length - 1]
            if source_last >= dest_first:
                break  # later starts in this run are no earlier; stop scanning it
            if _age_hours(dest_first, source_first) >= params.min_age_hours:
                candidates.append((run_index, start, source_first))
    candidates.sort(key=lambda item: (item[2], item[0], item[1]))
    return candidates


def apply_replay(
    request: AttackRequest,
    params: ReplayParameters,
    rng: np.random.Generator,
    settings: AttackSettings,
) -> AttackResult:
    clean = base.clean_array(request)
    if not np.isfinite(clean).all():
        return base.rejected_result(
            request,
            AttackFamily.REPLAY,
            AttackRejectionReason.MISSING_TARGET,
            "destination has NaN",
        )

    candidates = _source_candidates(request, params)
    if not candidates:
        return base.rejected_result(
            request,
            AttackFamily.REPLAY,
            AttackRejectionReason.NO_REPLAY_SOURCE,
            "no earlier finite source window of equal length in the same cell",
        )

    run_index, start, source_first = candidates[int(rng.integers(len(candidates)))]
    history = request.history
    assert history is not None  # guaranteed by non-empty candidates
    length = request.length
    source_values = np.asarray(
        history.run_values[run_index][start : start + length], dtype="float64"
    )
    source_labels = history.run_labels[run_index][start : start + length]
    source_last = history.run_times[run_index][start + length - 1]

    final, changed, saturated = base.finalize_candidate(clean, source_values, settings)
    age = _age_hours(request.window.first_observation_time, source_first)
    if not changed.any():
        return base.rejected_result(
            request,
            AttackFamily.REPLAY,
            AttackRejectionReason.ZERO_EFFECT,
            "replayed values equal the destination at every position",
        )

    source = ReplaySource(
        first_time=source_first,
        last_time=source_last,
        age_hours=age,
        row_labels=tuple(source_labels),
        candidate_count=len(candidates),
    )
    return base.build_result(
        request,
        AttackFamily.REPLAY,
        clean,
        final,
        changed,
        saturated,
        requested_magnitude=0.0,
        sign=0,
        extra={"age_hours": age, "source_candidates": len(candidates)},
        replay_source=source,
    )
