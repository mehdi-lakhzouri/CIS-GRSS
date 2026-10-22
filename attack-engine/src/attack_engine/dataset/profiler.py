"""Descriptive, JSON-compatible profile of an observation table.

Statistics are descriptive only: they summarise what the file contains and make
no claim about anomalies or attacks. The input DataFrame is never modified, and
no individual observations are included in the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from attack_engine.dataset import temporal
from attack_engine.dataset.schema import DatasetSchema
from attack_engine.dataset.temporal import TemporalGap
from attack_engine.dataset.validator import ValidationReport


class ProfilingError(ValueError):
    """The table lacks the columns needed to build a profile."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class ObservationPeriod(_Frozen):
    first_observation_end: datetime
    last_observation_end: datetime
    first_interval_start: datetime
    timestamp_convention: str = "interval_end_utc"


class TargetStatistics(_Frozen):
    column: str
    unit: str | None
    row_count: int
    finite_count: int
    missing_count: int
    coverage_percent: float
    minimum: float | None
    maximum: float | None
    mean: float | None
    # Sample standard deviation (ddof=1) of finite values.
    std: float | None


class StationProfile(_Frozen):
    row_count: int
    first_observation_end: datetime | None
    last_observation_end: datetime | None
    target_missing_count: int
    target_coverage_percent: float
    # Only columns with at least one missing value are listed.
    missing_by_column: dict[str, int]
    gap_count: int
    missing_intervals: int


class GapSummary(_Frozen):
    gap_count: int
    missing_intervals: int
    largest: tuple[TemporalGap, ...]


class ValidationSummary(_Frozen):
    passed: bool
    error_count: int
    warning_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


class DatasetProfile(_Frozen):
    schema_name: str
    row_count: int
    column_count: int
    station_count: int
    observation_period: ObservationPeriod | None
    stations: dict[str, StationProfile]
    missing_by_column: dict[str, int]
    target: TargetStatistics
    duplicate_key_count: int
    temporal_gaps: GapSummary
    year_boundary_row_count: int
    partition_row_counts: dict[str, int]
    validation: ValidationSummary | None


def _percent(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _target_statistics(values: np.ndarray, column: str, unit: str | None) -> TargetStatistics:
    finite = values[np.isfinite(values)]
    has_values = finite.size > 0
    return TargetStatistics(
        column=column,
        unit=unit,
        row_count=int(values.size),
        finite_count=int(finite.size),
        missing_count=int(np.isnan(values).sum()),
        coverage_percent=_percent(int(finite.size), int(values.size)),
        minimum=float(finite.min()) if has_values else None,
        maximum=float(finite.max()) if has_values else None,
        mean=float(finite.mean()) if has_values else None,
        std=float(finite.std(ddof=1)) if finite.size > 1 else None,
    )


def _summarize_validation(report: ValidationReport) -> ValidationSummary:
    return ValidationSummary(
        passed=report.passed,
        error_count=len(report.errors),
        warning_count=len(report.warnings),
        errors=tuple(f"{issue.code}: {issue.message}" for issue in report.errors),
        warnings=tuple(f"{issue.code}: {issue.message}" for issue in report.warnings),
    )


@dataclass(frozen=True)
class DatasetProfiler:
    schema: DatasetSchema
    max_listed_gaps: int = 10

    def profile(
        self, frame: pd.DataFrame, report: ValidationReport | None = None
    ) -> DatasetProfile:
        schema = self.schema
        time_column = schema.timestamps.column
        required = [schema.station_column, time_column, schema.target_column]
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ProfilingError(f"cannot profile dataset without columns {missing}")

        interval = schema.timestamps.interval
        stations = frame[schema.station_column]
        times_utc = temporal.to_utc(frame[time_column])
        target_values = frame[schema.target_column].to_numpy(dtype="float64", na_value=np.nan)
        steps = temporal.analyze_steps(stations, times_utc, interval)
        years = temporal.observation_year(times_utc, interval)

        return DatasetProfile(
            schema_name=schema.name,
            row_count=len(frame.index),
            column_count=len(frame.columns),
            station_count=int(stations.nunique()),
            observation_period=self._period(times_utc),
            stations=self._stations(frame, stations, times_utc, target_values, steps.gaps),
            missing_by_column={str(name): int(n) for name, n in frame.isna().sum().items()},
            target=_target_statistics(
                target_values, schema.target_column, schema.column(schema.target_column).unit
            ),
            duplicate_key_count=int(frame.duplicated(subset=list(schema.key_columns)).sum()),
            temporal_gaps=GapSummary(
                gap_count=len(steps.gaps),
                missing_intervals=steps.missing_interval_count,
                largest=tuple(
                    sorted(steps.gaps, key=lambda gap: gap.missing_intervals, reverse=True)[
                        : self.max_listed_gaps
                    ]
                ),
            ),
            year_boundary_row_count=int((times_utc.dt.year != years)[times_utc.notna()].sum()),
            partition_row_counts=dict(report.partition_row_counts) if report else {},
            validation=_summarize_validation(report) if report else None,
        )

    def _period(self, times_utc: pd.Series) -> ObservationPeriod | None:
        valid = times_utc.dropna()
        if valid.empty:
            return None
        first, last = valid.min(), valid.max()
        return ObservationPeriod(
            first_observation_end=first.to_pydatetime(),
            last_observation_end=last.to_pydatetime(),
            first_interval_start=(first - self.schema.timestamps.interval).to_pydatetime(),
        )

    def _stations(
        self,
        frame: pd.DataFrame,
        stations: pd.Series,
        times_utc: pd.Series,
        target_values: np.ndarray,
        gaps: tuple[TemporalGap, ...],
    ) -> dict[str, StationProfile]:
        missing = frame.isna().groupby(stations).sum()
        target_finite = pd.Series(np.isfinite(target_values), index=frame.index)
        profiles = {}
        for station, rows in stations.groupby(stations).groups.items():
            station_times = times_utc.loc[rows].dropna()
            station_gaps = [gap for gap in gaps if gap.station == str(station)]
            row_count = len(rows)
            finite = int(target_finite.loc[rows].sum())
            missing_row = missing.loc[station]
            profiles[str(station)] = StationProfile(
                row_count=row_count,
                first_observation_end=(
                    station_times.min().to_pydatetime() if not station_times.empty else None
                ),
                last_observation_end=(
                    station_times.max().to_pydatetime() if not station_times.empty else None
                ),
                target_missing_count=int(missing_row[self.schema.target_column]),
                target_coverage_percent=_percent(finite, row_count),
                missing_by_column={
                    str(name): int(n) for name, n in missing_row.items() if int(n) > 0
                },
                gap_count=len(station_gaps),
                missing_intervals=sum(gap.missing_intervals for gap in station_gaps),
            )
        return profiles
