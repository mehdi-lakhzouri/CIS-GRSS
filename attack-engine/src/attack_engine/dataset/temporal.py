"""Interval-end timestamp semantics for hourly observations.

A record stamped ``t`` covers ``[t - interval, t)``. Internally we derive
``interval_start = t - interval`` and ``observation_year = year(interval_start)``,
so the record stamped 2026-01-01T00:00Z closes the last hour of 2025.

Every helper returns new objects; the source timestamp column is never modified.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pydantic import BaseModel, ConfigDict

from attack_engine.dataset.schema import PartitionPlan

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


class TemporalGap(BaseModel):
    """Missing intervals between two consecutive records of one station."""

    model_config = ConfigDict(frozen=True)

    station: str
    last_before_gap: datetime
    first_after_gap: datetime
    missing_intervals: int


class StepAnalysis(BaseModel):
    """Classification of the steps between consecutive records of each station."""

    model_config = ConfigDict(frozen=True)

    gaps: tuple[TemporalGap, ...]
    # Positive steps that are not a whole number of intervals.
    irregular_step_count: int
    # Timestamps that do not fall on the interval grid (e.g. 01:30 for hourly data).
    misaligned_count: int

    @property
    def missing_interval_count(self) -> int:
        return sum(gap.missing_intervals for gap in self.gaps)


def is_timezone_naive(times: pd.Series) -> bool:
    return is_datetime64_any_dtype(times.dtype) and times.dt.tz is None


def to_utc(times: pd.Series) -> pd.Series:
    """Return a timezone-aware UTC copy of ``times``.

    Naive datetimes are declared UTC (the documented semantics of the source),
    aware datetimes are converted, and anything else is parsed as ISO 8601 with
    unparseable values becoming ``NaT``.
    """
    if is_datetime64_any_dtype(times.dtype):
        if times.dt.tz is None:
            return times.dt.tz_localize("UTC")
        return times.dt.tz_convert("UTC")
    return pd.to_datetime(times, errors="coerce", utc=True, format="ISO8601")


def interval_start(times_utc: pd.Series, interval: timedelta) -> pd.Series:
    return times_utc - interval


def observation_year(times_utc: pd.Series, interval: timedelta) -> pd.Series:
    return interval_start(times_utc, interval).dt.year


def intervals_in_year(year: int, interval: timedelta) -> int:
    """Number of whole intervals in a calendar year (8784 hours in a leap year)."""
    span = datetime(year + 1, 1, 1, tzinfo=UTC) - datetime(year, 1, 1, tzinfo=UTC)
    return span // interval


def analyze_steps(stations: pd.Series, times_utc: pd.Series, interval: timedelta) -> StepAnalysis:
    """Find gaps, irregular steps and off-grid timestamps per station.

    Zero-length steps (duplicate keys) are ignored here; they are reported by the
    key-uniqueness check.
    """
    frame = pd.DataFrame({"station": stations, "time": times_utc}).dropna()
    step_unit = pd.Timedelta(interval)
    misaligned = int(((frame["time"] - _EPOCH) % step_unit != pd.Timedelta(0)).sum())

    frame = frame.sort_values(["station", "time"], kind="stable")
    previous = frame.groupby("station", sort=False)["time"].shift()
    steps = frame["time"] - previous
    positive = steps.notna() & (steps > pd.Timedelta(0))
    on_grid = (steps % step_unit) == pd.Timedelta(0)
    irregular = positive & ~on_grid
    gap_mask = positive & on_grid & (steps > step_unit)

    gaps = tuple(
        TemporalGap(
            station=str(station),
            last_before_gap=before.to_pydatetime(),
            first_after_gap=after.to_pydatetime(),
            missing_intervals=int(step // step_unit) - 1,
        )
        for station, before, after, step in zip(
            frame.loc[gap_mask, "station"],
            previous[gap_mask],
            frame.loc[gap_mask, "time"],
            steps[gap_mask],
            strict=True,
        )
    )
    return StepAnalysis(
        gaps=gaps, irregular_step_count=int(irregular.sum()), misaligned_count=misaligned
    )


def assign_partitions(times_utc: pd.Series, plan: PartitionPlan, interval: timedelta) -> pd.Series:
    """Label each record with the partition containing its measured interval, or ``None``."""
    starts = interval_start(times_utc, interval)
    labels = np.full(len(starts), None, dtype=object)
    for partition in plan.partitions:
        lower = pd.Timestamp(partition.start).tz_convert("UTC")
        upper = pd.Timestamp(partition.end).tz_convert("UTC")
        labels[((starts >= lower) & (starts < upper)).to_numpy()] = partition.name
    return pd.Series(labels, index=times_utc.index, dtype="object")
