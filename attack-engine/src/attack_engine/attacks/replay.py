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

from attack_engine.attacks import base, multitarget
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


def _source_candidates(
    request: AttackRequest, params: ReplayParameters, settings: AttackSettings
) -> list[_Candidate]:
    history = request.history
    length = request.length
    dest_first = request.window.first_observation_time
    candidates: list[_Candidate] = []
    if history is None:
        return candidates
    for run_index, times in enumerate(history.run_times):
        valid_starts = None
        if params.target_columns != ("SOIL_MOISTURE_5",):
            valid_rows = np.ones(len(times), dtype=bool)
            for index, column in enumerate(params.target_columns):
                runs = history.run_values if index == 0 else history.run_secondary.get(column)
                if runs is None:
                    valid_rows[:] = False
                    break
                values = np.asarray(runs[run_index])
                bounds = multitarget.settings_for(column, settings).bounds
                valid_rows &= np.isfinite(values) & (values >= bounds.low) & (values <= bounds.high)
            invalid_prefix = np.r_[0, np.cumsum(~valid_rows)]
            valid_starts = invalid_prefix[length:] == invalid_prefix[:-length]
        for start in range(len(times) - length + 1):
            source_first = times[start]
            source_last = times[start + length - 1]
            if source_last >= dest_first:
                break  # later starts in this run are no earlier; stop scanning it
            if _age_hours(dest_first, source_first) >= params.min_age_hours:
                if valid_starts is not None:
                    if not valid_starts[start]:
                        continue
                    source_times = times[start : start + length]
                    elapsed = tuple(
                        (time - source_first).total_seconds() / 3600 for time in source_times
                    )
                    if elapsed != request.elapsed_hours:
                        continue
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

    candidates = _source_candidates(request, params, settings)
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

    age = _age_hours(request.window.first_observation_time, source_first)
    source = ReplaySource(
        first_time=source_first,
        last_time=source_last,
        age_hours=age,
        row_labels=tuple(source_labels),
        candidate_count=len(candidates),
    )

    # Primary plus any selected secondary variable: replay the same source block, time-aligned.
    column_candidates: list[tuple[str, np.ndarray, np.ndarray]] = [
        (params.target_columns[0], clean, source_values)
    ]
    for column in params.target_columns[1:]:
        run = history.run_secondary.get(column)
        dest_clean = multitarget.secondary_clean_array(request, column)
        assert run is not None  # all selected channels validated before source selection
        candidate = np.asarray(run[run_index][start : start + length], dtype="float64")
        candidate = np.where(np.isfinite(dest_clean), candidate, dest_clean)
        column_candidates.append((column, dest_clean, candidate))

    return multitarget.build_multi_result(
        request,
        AttackFamily.REPLAY,
        column_candidates,
        settings,
        requested_magnitude=0.0,
        sign=0,
        extra={"age_hours": age, "source_candidates": len(candidates)},
        replay_source=source,
    )
