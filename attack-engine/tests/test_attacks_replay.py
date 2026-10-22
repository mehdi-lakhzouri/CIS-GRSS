from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import pytest

from attack_engine.attacks.config import ReplayParameters
from attack_engine.attacks.models import (
    AttackRejectionReason,
    AttackRequest,
    AttackSettings,
    ReplayHistory,
    TargetBounds,
)
from attack_engine.attacks.replay import apply_replay
from attack_engine.scheduling.models import AttackFamily

RequestFactory = Callable[..., AttackRequest]
WindowFactory = Callable[..., object]
SETTINGS = AttackSettings(bounds=TargetBounds(0.0, 0.70), quantum=None)


@pytest.mark.parametrize("quantum", [None, 0.001])
def test_replay_preserves_identical_positions_exactly(
    make_attack_window: WindowFactory, make_attack_request: RequestFactory, quantum: float | None
) -> None:
    window = make_attack_window(AttackFamily.REPLAY, start="2024-03-10 00:00", length=3)
    clean = [0.141, 0.1414, 0.3]
    history = _history("2024-03-01 00:00", [0.141, 0.1414, 0.4], [0, 1, 2])
    request = make_attack_request(AttackFamily.REPLAY, clean, window=window, history=history)
    result = apply_replay(
        request, ReplayParameters(min_age_hours=1), _rng(), AttackSettings(quantum=quantum)
    )
    assert result.applied
    assert result.changed_mask == (False, False, True)
    assert result.candidate_values[:2] == request.clean_values[:2]


def test_replay_preserves_a_real_sub_nanounit_difference(
    make_attack_window: WindowFactory, make_attack_request: RequestFactory
) -> None:
    window = make_attack_window(AttackFamily.REPLAY, start="2024-03-10 00:00", length=1)
    source = 0.141 + 1e-12
    history = _history("2024-03-01 00:00", [source], [0])
    request = make_attack_request(AttackFamily.REPLAY, [0.141], window=window, history=history)
    result = apply_replay(request, ReplayParameters(min_age_hours=1), _rng(), SETTINGS)
    assert result.applied
    assert result.changed_mask == (True,)
    assert result.candidate_values == (source,)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _history(start: str, values: Sequence[float], labels: Sequence[int]) -> ReplayHistory:
    times = tuple(
        ts.to_pydatetime() for ts in pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    )
    return ReplayHistory(
        run_times=(times,),
        run_values=(tuple(float(v) for v in values),),
        run_labels=(tuple(labels),),
    )


def test_replay_copies_earlier_source(
    make_attack_window: WindowFactory, make_attack_request: RequestFactory
) -> None:
    window = make_attack_window(
        AttackFamily.REPLAY, start="2024-03-10 00:00", length=3, labels=[100, 101, 102]
    )
    history = _history("2024-03-01 00:00", [0.10, 0.11, 0.12, 0.13, 0.14], [0, 1, 2, 3, 4])
    request = make_attack_request(
        AttackFamily.REPLAY, [0.30, 0.31, 0.32], window=window, history=history
    )

    result = apply_replay(request, ReplayParameters(min_age_hours=1), _rng(), SETTINGS)

    assert result.applied
    assert result.modified_count == 3
    assert result.replay_source is not None
    assert result.replay_source.last_time < window.first_observation_time
    # candidate values come from the source run
    assert all(value in {0.10, 0.11, 0.12, 0.13, 0.14} for value in result.candidate_values)


def test_replay_no_source_when_history_is_later(
    make_attack_window: WindowFactory, make_attack_request: RequestFactory
) -> None:
    window = make_attack_window(AttackFamily.REPLAY, start="2024-03-01 00:00", length=2)
    history = _history("2024-03-05 00:00", [0.1, 0.2, 0.3], [10, 11, 12])
    request = make_attack_request(AttackFamily.REPLAY, [0.3, 0.4], window=window, history=history)

    result = apply_replay(request, ReplayParameters(min_age_hours=1), _rng(), SETTINGS)
    assert result.rejection_reason is AttackRejectionReason.NO_REPLAY_SOURCE


def test_replay_no_source_when_history_absent(make_attack_request: RequestFactory) -> None:
    request = make_attack_request(AttackFamily.REPLAY, [0.3, 0.4], history=None)
    result = apply_replay(request, ReplayParameters(), _rng(), SETTINGS)
    assert result.rejection_reason is AttackRejectionReason.NO_REPLAY_SOURCE


def test_replay_zero_effect_when_source_equals_destination(
    make_attack_window: WindowFactory, make_attack_request: RequestFactory
) -> None:
    window = make_attack_window(AttackFamily.REPLAY, start="2024-03-10 00:00", length=2)
    history = _history("2024-03-01 00:00", [0.30, 0.31], [0, 1])
    request = make_attack_request(AttackFamily.REPLAY, [0.30, 0.31], window=window, history=history)
    result = apply_replay(request, ReplayParameters(min_age_hours=1), _rng(), SETTINGS)
    assert result.rejection_reason is AttackRejectionReason.ZERO_EFFECT


def test_replay_respects_min_age(
    make_attack_window: WindowFactory, make_attack_request: RequestFactory
) -> None:
    window = make_attack_window(AttackFamily.REPLAY, start="2024-03-01 10:00", length=2)
    # Source starts only 2 h before the window; min_age 48 h rejects it.
    history = _history("2024-03-01 08:00", [0.1, 0.2], [0, 1])
    request = make_attack_request(AttackFamily.REPLAY, [0.3, 0.4], window=window, history=history)
    result = apply_replay(request, ReplayParameters(min_age_hours=48), _rng(), SETTINGS)
    assert result.rejection_reason is AttackRejectionReason.NO_REPLAY_SOURCE


def test_replay_records_age_and_source_labels(
    make_attack_window: WindowFactory, make_attack_request: RequestFactory
) -> None:
    window = make_attack_window(AttackFamily.REPLAY, start="2024-03-02 00:00", length=2)
    history = _history("2024-03-01 00:00", [0.10, 0.11, 0.12], [7, 8, 9])
    request = make_attack_request(AttackFamily.REPLAY, [0.3, 0.4], window=window, history=history)

    result = apply_replay(request, ReplayParameters(min_age_hours=1), _rng(), SETTINGS)
    source = result.replay_source
    assert source is not None
    assert source.age_hours >= 1
    assert len(source.row_labels) == 2
    assert set(source.row_labels) <= {7, 8, 9}


def test_replay_is_deterministic(
    make_attack_window: WindowFactory, make_attack_request: RequestFactory
) -> None:
    window = make_attack_window(AttackFamily.REPLAY, start="2024-03-10 00:00", length=2)
    history = _history("2024-03-01 00:00", [0.10, 0.11, 0.12, 0.13, 0.14, 0.15], list(range(6)))
    request = make_attack_request(AttackFamily.REPLAY, [0.3, 0.4], window=window, history=history)
    first = apply_replay(request, ReplayParameters(min_age_hours=1), _rng(9), SETTINGS)
    second = apply_replay(request, ReplayParameters(min_age_hours=1), _rng(9), SETTINGS)
    assert first.candidate_values == second.candidate_values
    assert first.replay_source == second.replay_source


def test_replay_rejects_missing_destination(
    make_attack_window: WindowFactory, make_attack_request: RequestFactory
) -> None:
    window = make_attack_window(AttackFamily.REPLAY, start="2024-03-10 00:00", length=2)
    history = _history("2024-03-01 00:00", [0.1, 0.2], [0, 1])
    request = make_attack_request(
        AttackFamily.REPLAY, [0.3, np.nan], window=window, history=history
    )
    result = apply_replay(request, ReplayParameters(min_age_hours=1), _rng(), SETTINGS)
    assert result.rejection_reason is AttackRejectionReason.MISSING_TARGET
