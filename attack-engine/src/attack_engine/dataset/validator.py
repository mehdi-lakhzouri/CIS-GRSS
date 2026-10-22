"""Strict, read-only validation of an observation table.

The validator reports problems; it never corrects, drops or reorders data. All
derived values (UTC timestamps, interval starts, observation years) are computed
on copies.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from pydantic import BaseModel, ConfigDict

from attack_engine.dataset import temporal
from attack_engine.dataset.schema import (
    NUMERIC_KINDS,
    ColumnKind,
    DatasetSchema,
    PartitionPlan,
    ValidationProfile,
)

_MAX_LISTED = 5


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: Severity
    message: str
    column: str | None = None
    count: int | None = None


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_name: str
    profile_name: str
    row_count: int
    column_count: int
    issues: tuple[ValidationIssue, ...]
    partition_row_counts: dict[str, int] = {}

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    def codes(self, severity: Severity | None = None) -> set[str]:
        return {
            issue.code for issue in self.issues if severity is None or issue.severity is severity
        }


def _issue(
    severity: Severity, code: str, message: str, column: str | None = None, count: int | None = None
) -> ValidationIssue:
    return ValidationIssue(
        code=code, severity=severity, message=message, column=column, count=count
    )


def _listing(values: Iterable[object]) -> str:
    items = list(values)
    shown = ", ".join(repr(item) for item in items[:_MAX_LISTED])
    return shown + (f", ... ({len(items)} total)" if len(items) > _MAX_LISTED else "")


def _is_compatible(series: pd.Series, kind: ColumnKind) -> bool:
    dtype = series.dtype
    if kind is ColumnKind.DATETIME:
        return ptypes.is_datetime64_any_dtype(dtype)
    if kind is ColumnKind.BOOLEAN:
        return ptypes.is_bool_dtype(dtype)
    if ptypes.is_bool_dtype(dtype):
        return False
    if kind is ColumnKind.INTEGER:
        return ptypes.is_integer_dtype(dtype)
    if kind is ColumnKind.FLOAT:
        return ptypes.is_float_dtype(dtype) or ptypes.is_integer_dtype(dtype)
    return ptypes.is_string_dtype(series)


@dataclass(frozen=True)
class DatasetValidator:
    schema: DatasetSchema
    profile: ValidationProfile
    partitions: PartitionPlan | None = None

    def validate(self, frame: pd.DataFrame) -> ValidationReport:
        issues: list[ValidationIssue] = []
        partition_counts: dict[str, int] = {}
        if len(frame.index) == 0:
            issues.append(_issue(Severity.ERROR, "empty_dataset", "Dataset contains no rows."))
        elif frame.columns.duplicated().any():
            duplicated = frame.columns[frame.columns.duplicated()].unique()
            issues.append(
                _issue(
                    Severity.ERROR,
                    "duplicate_column_name",
                    f"Duplicate column names: {_listing(duplicated)}.",
                )
            )
        else:
            partition_counts = self._run_checks(frame, issues)
        return ValidationReport(
            schema_name=self.schema.name,
            profile_name=self.profile.name,
            row_count=len(frame.index),
            column_count=len(frame.columns),
            issues=tuple(issues),
            partition_row_counts=partition_counts,
        )

    def _run_checks(self, frame: pd.DataFrame, issues: list[ValidationIssue]) -> dict[str, int]:
        schema = self.schema
        issues += self._check_columns(frame)
        issues += self._check_dimensions(frame)
        compatible = {
            spec.name
            for spec in schema.columns
            if spec.name in frame.columns and _is_compatible(frame[spec.name], spec.kind)
        }
        issues += self._check_types(frame, compatible)
        issues += self._check_nulls(frame)
        issues += self._check_numeric_values(frame, compatible)

        if schema.station_column in compatible:
            issues += self._check_stations(frame, compatible)
        if all(name in frame.columns for name in schema.key_columns):
            duplicates = int(frame.duplicated(subset=list(schema.key_columns)).sum())
            if duplicates:
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "duplicate_observation_key",
                        f"{duplicates} rows repeat an existing {schema.key_columns} key.",
                        count=duplicates,
                    )
                )

        if schema.timestamps.column not in frame.columns:
            return {}
        times_utc = temporal.to_utc(frame[schema.timestamps.column])
        issues += self._check_timestamps(frame, times_utc, compatible)
        if self.partitions is None:
            return {}
        partition_issues, counts = self._check_partitions(times_utc, self.partitions)
        issues += partition_issues
        return counts

    # Structural checks -------------------------------------------------------------

    def _check_columns(self, frame: pd.DataFrame) -> list[ValidationIssue]:
        issues = []
        for spec in self.schema.columns:
            if spec.required and spec.name not in frame.columns:
                is_target = spec.name == self.schema.target_column
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "missing_target_column" if is_target else "missing_required_column",
                        f"Required column {spec.name!r} is missing.",
                        column=spec.name,
                    )
                )
        unexpected = [str(name) for name in frame.columns if name not in self.schema.column_names]
        if unexpected:
            issues.append(
                _issue(
                    Severity.WARNING,
                    "unexpected_column",
                    f"Columns not described by the schema: {_listing(unexpected)}.",
                    count=len(unexpected),
                )
            )
        return issues

    def _check_dimensions(self, frame: pd.DataFrame) -> list[ValidationIssue]:
        issues = []
        expected_rows = self.profile.expected_row_count
        if expected_rows is not None and len(frame.index) != expected_rows:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "row_count_mismatch",
                    f"Expected {expected_rows} rows, found {len(frame.index)}.",
                    count=len(frame.index),
                )
            )
        expected_columns = self.profile.expected_column_count
        if expected_columns is not None and len(frame.columns) != expected_columns:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "column_count_mismatch",
                    f"Expected {expected_columns} columns, found {len(frame.columns)}.",
                    count=len(frame.columns),
                )
            )
        return issues

    def _check_types(self, frame: pd.DataFrame, compatible: set[str]) -> list[ValidationIssue]:
        return [
            _issue(
                Severity.ERROR,
                "incompatible_type",
                f"Column {spec.name!r} has dtype {frame[spec.name].dtype}, "
                f"expected {spec.kind.value}.",
                column=spec.name,
            )
            for spec in self.schema.columns
            if spec.name in frame.columns and spec.name not in compatible
        ]

    def _check_nulls(self, frame: pd.DataFrame) -> list[ValidationIssue]:
        issues = []
        for spec in self.schema.columns:
            if spec.nullable or spec.name not in frame.columns:
                continue
            nulls = int(frame[spec.name].isna().sum())
            if nulls:
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "null_in_mandatory_column",
                        f"Mandatory column {spec.name!r} contains {nulls} null values.",
                        column=spec.name,
                        count=nulls,
                    )
                )
        return issues

    # Numerical checks --------------------------------------------------------------

    def _check_numeric_values(
        self, frame: pd.DataFrame, compatible: set[str]
    ) -> list[ValidationIssue]:
        issues = []
        for spec in self.schema.columns:
            if spec.kind not in NUMERIC_KINDS or spec.name not in compatible:
                continue
            values = frame[spec.name].to_numpy(dtype="float64", na_value=np.nan)
            non_finite = int(np.isinf(values).sum())
            if non_finite:
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "non_finite_value",
                        f"Column {spec.name!r} contains {non_finite} infinite values.",
                        column=spec.name,
                        count=non_finite,
                    )
                )
            finite = values[np.isfinite(values)]
            low = spec.min_value if spec.min_value is not None else -np.inf
            high = spec.max_value if spec.max_value is not None else np.inf
            outside = finite[(finite < low) | (finite > high)]
            if outside.size:
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "out_of_range",
                        f"Column {spec.name!r} has {outside.size} values outside "
                        f"[{low}, {high}] (observed {outside.min()} to {outside.max()}).",
                        column=spec.name,
                        count=int(outside.size),
                    )
                )

        target = self.schema.target_column
        if target in frame.columns:
            missing = int(frame[target].isna().sum())
            if missing:
                issues.append(
                    _issue(
                        Severity.INFO,
                        "target_missing_values",
                        f"Target {target!r} has {missing} missing observations (allowed, "
                        "kept as missing).",
                        column=target,
                        count=missing,
                    )
                )
        return issues

    # Station checks ----------------------------------------------------------------

    def _check_stations(self, frame: pd.DataFrame, compatible: set[str]) -> list[ValidationIssue]:
        issues = []
        column = self.schema.station_column
        stations = frame[column].dropna().astype(str)
        observed = set(stations.unique())

        pattern = re.compile(self.schema.station_id_pattern)
        invalid = sorted(station for station in observed if not pattern.fullmatch(station))
        if invalid:
            rows = int(stations.isin(invalid).sum())
            issues.append(
                _issue(
                    Severity.ERROR,
                    "invalid_station_id",
                    f"{rows} rows carry malformed station identifiers: {_listing(invalid)}.",
                    column=column,
                    count=rows,
                )
            )

        if self.profile.expected_stations is not None:
            expected = set(self.profile.expected_stations)
            unknown = sorted(observed - expected - set(invalid))
            if unknown:
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "unknown_station",
                        f"Stations not in the expected inventory: {_listing(unknown)}.",
                        column=column,
                        count=len(unknown),
                    )
                )
            absent = sorted(expected - observed)
            if absent:
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "missing_station",
                        f"Expected stations absent from the dataset: {_listing(absent)}.",
                        column=column,
                        count=len(absent),
                    )
                )

        coordinates = [self.schema.latitude_column, self.schema.longitude_column]
        if all(name in compatible for name in coordinates):
            distinct = frame.groupby(column)[coordinates].nunique()
            inconsistent = list(distinct.index[(distinct > 1).any(axis=1)])
            if inconsistent:
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "inconsistent_station_coordinates",
                        f"Stations with more than one coordinate pair: {_listing(inconsistent)}.",
                        column=column,
                        count=len(inconsistent),
                    )
                )
        return issues

    # Temporal checks ---------------------------------------------------------------

    def _check_timestamps(
        self, frame: pd.DataFrame, times_utc: pd.Series, compatible: set[str]
    ) -> list[ValidationIssue]:
        issues = []
        semantics = self.schema.timestamps
        source = frame[semantics.column]

        unparseable = int((source.notna() & times_utc.isna()).sum())
        if unparseable:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "unparseable_timestamp",
                    f"{unparseable} timestamps cannot be parsed.",
                    column=semantics.column,
                    count=unparseable,
                )
            )
        if temporal.is_timezone_naive(source):
            issues.append(
                _issue(
                    Severity.INFO,
                    "naive_timestamps_interpreted_as_utc",
                    f"Column {semantics.column!r} is timezone-naive; values are interpreted as "
                    "UTC interval ends (source column left unchanged).",
                    column=semantics.column,
                )
            )

        years = temporal.observation_year(times_utc, semantics.interval)
        issues += self._check_year_column(frame, times_utc, years, compatible)

        station_column = self.schema.station_column
        if station_column in frame.columns:
            stations = frame[station_column]
            issues += self._check_ordering(stations, times_utc)
            issues += self._check_continuity(stations, times_utc)
            issues += self._check_year_coverage(stations, years)
        return issues

    def _check_year_column(
        self,
        frame: pd.DataFrame,
        times_utc: pd.Series,
        years: pd.Series,
        compatible: set[str],
    ) -> list[ValidationIssue]:
        issues = []
        valid = times_utc.notna()
        boundary_rows = int((times_utc.dt.year != years)[valid].sum())
        if boundary_rows:
            issues.append(
                _issue(
                    Severity.INFO,
                    "year_boundary_rows",
                    f"{boundary_rows} rows are stamped at the first instant of a year and close "
                    "the final interval of the previous observation year.",
                    count=boundary_rows,
                )
            )
        year_column = self.schema.timestamps.year_column
        if year_column is None or year_column not in compatible:
            return issues
        declared = frame[year_column]
        comparable = valid & declared.notna()
        mismatched = int((declared[comparable] != years[comparable]).sum())
        if mismatched:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "observation_year_mismatch",
                    f"{mismatched} rows have {year_column!r} different from year(time - interval).",
                    column=year_column,
                    count=mismatched,
                )
            )
        return issues

    def _check_ordering(self, stations: pd.Series, times_utc: pd.Series) -> list[ValidationIssue]:
        valid = times_utc.notna() & stations.notna()
        monotonic = (
            times_utc[valid]
            .groupby(stations[valid])
            .apply(lambda series: series.is_monotonic_increasing)
        )
        unordered = sorted(str(station) for station in monotonic.index[~monotonic.astype(bool)])
        if not unordered:
            return []
        return [
            _issue(
                Severity.WARNING,
                "unordered_timestamps",
                f"Rows are not in chronological order for stations: {_listing(unordered)} "
                "(order left unchanged).",
                count=len(unordered),
            )
        ]

    def _check_continuity(self, stations: pd.Series, times_utc: pd.Series) -> list[ValidationIssue]:
        issues = []
        steps = temporal.analyze_steps(stations, times_utc, self.schema.timestamps.interval)
        if steps.misaligned_count:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "misaligned_timestamp",
                    f"{steps.misaligned_count} timestamps are not aligned to the "
                    f"{self.schema.timestamps.interval} grid.",
                    count=steps.misaligned_count,
                )
            )
        if steps.irregular_step_count:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "irregular_time_step",
                    f"{steps.irregular_step_count} consecutive-record steps are not a whole "
                    f"multiple of {self.schema.timestamps.interval}.",
                    count=steps.irregular_step_count,
                )
            )
        if steps.gaps:
            severity = Severity.WARNING if self.profile.allow_temporal_gaps else Severity.ERROR
            first = steps.gaps[0]
            issues.append(
                _issue(
                    severity,
                    "temporal_gap",
                    f"{len(steps.gaps)} gaps totalling {steps.missing_interval_count} missing "
                    f"intervals (first: {first.station} after "
                    f"{first.last_before_gap.isoformat()}).",
                    count=len(steps.gaps),
                )
            )
        return issues

    def _check_year_coverage(self, stations: pd.Series, years: pd.Series) -> list[ValidationIssue]:
        counts = (
            pd.DataFrame({"station": stations, "year": years})
            .dropna()
            .groupby(["station", "year"])
            .size()
        )
        interval = self.schema.timestamps.interval
        excess, incomplete = [], []
        for (station, year), count in counts.items():
            expected = temporal.intervals_in_year(int(year), interval)
            label = f"{station}/{int(year)}: {count} of {expected}"
            if count > expected:
                excess.append(label)
            elif count < expected:
                incomplete.append(label)

        issues = []
        if excess:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "excess_observations_in_year",
                    f"Station-years with more records than intervals: {_listing(excess)}.",
                    count=len(excess),
                )
            )
        if incomplete and self.profile.require_complete_years:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "incomplete_observation_year",
                    f"Station-years missing records: {_listing(incomplete)}.",
                    count=len(incomplete),
                )
            )
        return issues

    # Partition checks --------------------------------------------------------------

    def _check_partitions(
        self, times_utc: pd.Series, plan: PartitionPlan
    ) -> tuple[list[ValidationIssue], dict[str, int]]:
        labels = temporal.assign_partitions(times_utc, plan, self.schema.timestamps.interval)
        counts = {
            partition.name: int((labels == partition.name).sum()) for partition in plan.partitions
        }
        unassigned = int((labels.isna() & times_utc.notna()).sum())
        issues = []
        if unassigned:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "unassigned_partition",
                    f"{unassigned} records fall outside every configured partition.",
                    count=unassigned,
                )
            )
        return issues, counts
