"""Synthetic detector inputs spanning the end of train and the start of validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_processing.fit import fit_params
from signal_processing.params import FrozenParams
from signal_processing.partition import DEFAULT_RULES

STATIONS = ("ST_A", "ST_B")
START = pd.Timestamp("2023-09-01 01:00")
HOURS = 24 * 200  # ~4 months train, ~2.5 months validation


def synthetic_station(seed: int, hours: int = HOURS) -> pd.DataFrame:
    """Drydown with rain pulses, diurnal noise, sensor quantization and NaN gaps."""
    rng = np.random.default_rng(seed)
    rain = np.where(rng.random(hours) < 0.02, rng.gamma(1.5, 2.0, hours), 0.0).round(1)
    sm5, sm10 = np.empty(hours), np.empty(hours)
    a5 = a10 = 0.25
    for t in range(hours):
        a5 = 0.08 + (a5 - 0.08) * np.exp(-1 / 200) + 0.004 * rain[t]
        a10 = 0.10 + (a10 - 0.10) * np.exp(-1 / 300) + 0.6 * (a5 - a10) / 24
        sm5[t], sm10[t] = min(a5, 0.5), min(a10, 0.5)
    diurnal = 0.0015 * np.sin(2 * np.pi * np.arange(hours) / 24)
    sm5 = (sm5 + diurnal + rng.normal(0, 0.0007, hours)).round(3)
    sm10 = (sm10 + 0.5 * diurnal + rng.normal(0, 0.0005, hours)).round(3)
    for column, n_gaps in ((sm5, 6), (sm10, 4), (rain, 3)):
        for start in rng.integers(100, hours - 50, n_gaps):
            column[start : start + rng.integers(1, 12)] = np.nan
    return pd.DataFrame(
        {
            "time": START + pd.to_timedelta(np.arange(hours), unit="h"),
            "SOIL_MOISTURE_5": sm5,
            "SOIL_MOISTURE_10": sm10,
            "P_CALC": rain,
        }
    )


def make_frame(hours: int = HOURS) -> pd.DataFrame:
    parts = [synthetic_station(i, hours).assign(station=s) for i, s in enumerate(STATIONS)]
    frame = pd.concat(parts, ignore_index=True)
    return frame[["station", "time", "SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC"]]


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="session")
def params(frame: pd.DataFrame) -> FrozenParams:
    return fit_params(frame, "0" * 64, DEFAULT_RULES, "built-in defaults")
