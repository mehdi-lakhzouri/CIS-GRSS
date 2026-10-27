"""Causality, alignment, NaN propagation, gap resets and replay detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_processing.features import (
    FEATURE_COLUMNS,
    capped_hold,
    cusum,
    replay_flags,
    transform,
)
from signal_processing.params import Constants, FrozenParams

from .conftest import synthetic_station

REPLAY_FREE = [c for c in FEATURE_COLUMNS if not c.startswith("sp_replay")]


def test_causality_future_perturbation_leaves_past_unchanged(
    frame: pd.DataFrame, params: FrozenParams
) -> None:
    cut = pd.Timestamp("2024-01-20 12:00")
    future = frame["time"] > cut
    rng = np.random.default_rng(3)
    perturbed = frame.copy()
    for column in ("SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC"):
        perturbed.loc[future, column] = rng.uniform(0, 0.6, future.sum()).round(3)
    perturbed.loc[np.flatnonzero(future)[::5], "SOIL_MOISTURE_10"] = np.nan
    base = transform(frame, params)[~future.to_numpy()]
    moved = transform(perturbed, params)[~future.to_numpy()]
    pd.testing.assert_frame_equal(base, moved)


def test_alignment_same_rows_keys_and_order(frame: pd.DataFrame, params: FrozenParams) -> None:
    shuffled = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    out = transform(shuffled, params)
    assert len(out) == len(shuffled)
    pd.testing.assert_frame_equal(out[["station", "time"]], shuffled[["station", "time"]])
    ordered = transform(frame, params)
    merged = out.merge(ordered, on=["station", "time"], suffixes=("", "_ref"))
    for column in FEATURE_COLUMNS:
        np.testing.assert_array_equal(merged[column].to_numpy(), merged[f"{column}_ref"].to_numpy())


@pytest.mark.parametrize("depth", ["5", "10"])
def test_nan_input_gives_nan_features(
    frame: pd.DataFrame, params: FrozenParams, depth: str
) -> None:
    out = transform(frame, params)
    missing = frame[f"SOIL_MOISTURE_{depth}"].isna().to_numpy()
    assert missing.any()
    dependent = [c for c in REPLAY_FREE if c.endswith(f"_{depth}") or c.startswith("sp_depth")]
    for column in dependent:
        assert out.loc[missing, column].isna().all(), column
    assert (out.loc[missing, "sp_replay_match"] == 0).all()
    assert (out.loc[missing, "sp_replay_run"] == 0).all()


def test_features_mostly_defined(frame: pd.DataFrame, params: FrozenParams) -> None:
    out = transform(frame, params)
    for column in REPLAY_FREE:
        assert out[column].notna().mean() > 0.8, column


def test_cusum_resets_after_long_gap_and_holds_in_rain() -> None:
    z = np.array([3.0, 3.0, *([np.nan] * 7), 3.0, 3.0])
    out = cusum(z, np.zeros(len(z), dtype=bool), drift=0.5, gap_reset=6)
    assert out[1] == pytest.approx(5.0)
    assert np.isnan(out[2:9]).all()
    assert out[9] == pytest.approx(2.5)  # reset: 8 h since the last valid z
    held = cusum(np.array([3.0, 3.0, 3.0]), np.array([False, True, False]), 0.5, 6)
    assert held.tolist() == pytest.approx([2.5, 2.5, 5.0])


def test_rain_hold_is_capped() -> None:
    hold = np.array([False, True, True, True, True, False, True, True])
    assert capped_hold(hold, 2).tolist() == [False, True, True, False, False, False, True, True]


def test_short_gap_bridged_long_gap_resets(params: FrozenParams) -> None:
    station = synthetic_station(5, 600).assign(station="ST_A")
    station.loc[100:103, "SOIL_MOISTURE_5"] = np.nan  # 4 missing -> 5 h elapsed: bridged
    station.loc[300:309, "SOIL_MOISTURE_5"] = np.nan  # 10 missing -> 11 h elapsed: reset
    station.loc[:, "P_CALC"] = station["P_CALC"].fillna(0.0)
    out = transform(station, params)
    assert np.isfinite(out.loc[104, "sp_innov_5"])
    assert np.isnan(out.loc[310, "sp_innov_5"])
    assert np.isfinite(out.loc[311, "sp_innov_5"])


def test_replay_fires_on_copy_not_on_original() -> None:
    c = Constants()
    base = synthetic_station(9, 1500)
    sm5 = base["SOIL_MOISTURE_5"].to_numpy().copy()
    sm10 = base["SOIL_MOISTURE_10"].to_numpy().copy()
    rain = base["P_CALC"].to_numpy().copy()
    for x in (sm5, sm10, rain):
        x[np.isnan(x)] = 0.1  # keep the source block fully finite
    source, target, length = slice(200, 248), slice(1200, 1248), 48
    for x in (sm5, sm10, rain):
        x[target] = x[source]
    result = replay_flags(sm5, sm10, rain, c)
    w = c.replay_window_hours
    assert result.match[source].sum() == 0
    assert result.match[1200 + w - 1 : 1248].all()
    assert result.run[1247] == length - w + 1
    outside = np.ones(len(sm5), dtype=bool)
    outside[target] = False
    assert result.match[outside].sum() == 0


def test_replay_ignores_flat_windows() -> None:
    c = Constants()
    flat = np.full(200, 0.2)
    result = replay_flags(flat, flat.copy(), np.zeros(200), c)
    assert result.windows == 0
    assert result.match.sum() == 0
