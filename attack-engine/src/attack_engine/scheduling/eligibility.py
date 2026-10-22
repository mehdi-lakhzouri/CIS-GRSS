"""Identify valid attack opportunities without touching the source data.

The engine reads the loaded DataFrame through the existing temporal helpers and
reports, per (station, partition) cell:

- the finite-target denominator ``N_finite`` (before any family constraint);
- maximal continuous segments of hourly-consecutive, finite-target observations;
- why a cell yields no usable segment.

It never replaces missing values, never reorders or mutates the original
DataFrame, and does not itself draw or reserve windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from attack_engine.dataset import temporal
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.scheduling.models import RejectionReason, ScopeCell, WindowRejection


class EligibilityError(ValueError):
    """The dataset lacks the columns needed to assess eligibility."""


class Segment(BaseModel):
    """A maximal run of hourly-consecutive, finite-target observations in one cell."""

    model_config = ConfigDict(frozen=True)

    station: str
    partition: str
    row_labels: tuple[int, ...]
    times: tuple[datetime, ...]

    @property
    def length(self) -> int:
        return len(self.row_labels)

    @property
    def cell(self) -> ScopeCell:
        return ScopeCell(station=self.station, partition=self.partition)


class CellEligibility(BaseModel):
    model_config = ConfigDict(frozen=True)

    cell: ScopeCell
    row_count: int
    finite_target_count: int
    missing_target_count: int
    segments: tuple[Segment, ...]

    @property
    def eligible_observation_count(self) -> int:
        """Finite-target observations that also sit inside a continuous segment."""
        return sum(segment.length for segment in self.segments)


class EligibilityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    cells: tuple[CellEligibility, ...]
    rejections: tuple[WindowRejection, ...]

    def cell(self, cell: ScopeCell) -> CellEligibility | None:
        return next((item for item in self.cells if item.cell == cell), None)

    @property
    def finite_by_cell(self) -> dict[ScopeCell, int]:
        return {item.cell: item.finite_target_count for item in self.cells}


@dataclass(frozen=True)
class EligibilityEngine:
    schema: DatasetSchema
    partitions: PartitionPlan
    scope: tuple[str, ...]

    def analyze(self, frame: pd.DataFrame) -> EligibilityReport:
        schema = self.schema
        needed = [schema.station_column, schema.timestamps.column, schema.target_column]
        missing = [name for name in needed if name not in frame.columns]
        if missing:
            raise EligibilityError(f"cannot assess eligibility without columns {missing}")

        interval = schema.timestamps.interval
        # Build a sorted working view from copies; the source frame is untouched.
        times_utc = temporal.to_utc(frame[schema.timestamps.column])
        work = pd.DataFrame(
            {
                "label": np.asarray(frame.index),
                "station": frame[schema.station_column].to_numpy(),
                "time": times_utc.to_numpy(),
                "finite": np.isfinite(
                    frame[schema.target_column].to_numpy(dtype="float64", na_value=np.nan)
                ),
                "partition": temporal.assign_partitions(
                    times_utc, self.partitions, interval
                ).to_numpy(),
            }
        )
        work = work.sort_values(["station", "time"], kind="stable")

        cells: list[CellEligibility] = []
        rejections: list[WindowRejection] = []
        for partition in self.scope:
            partition_rows = work[work["partition"] == partition]
            for station in sorted(set(partition_rows["station"].tolist())):
                cell_rows = partition_rows[partition_rows["station"] == station]
                cells.append(self._build_cell(station, partition, cell_rows, interval, rejections))
        return EligibilityReport(cells=tuple(cells), rejections=tuple(rejections))

    def _build_cell(
        self,
        station: str,
        partition: str,
        rows: pd.DataFrame,
        interval: timedelta,
        rejections: list[WindowRejection],
    ) -> CellEligibility:
        segments = self._segments(station, partition, rows, interval)
        finite = int(rows["finite"].sum())
        cell = ScopeCell(station=station, partition=partition)
        if not segments:
            rejections.append(
                WindowRejection(
                    station=station,
                    partition=partition,
                    reason=RejectionReason.NO_ELIGIBLE_SEGMENT,
                    detail=f"{finite} finite targets but no continuous segment",
                )
            )
        return CellEligibility(
            cell=cell,
            row_count=len(rows),
            finite_target_count=finite,
            missing_target_count=int((~rows["finite"]).sum()),
            segments=tuple(segments),
        )

    def _segments(
        self, station: str, partition: str, rows: pd.DataFrame, interval: timedelta
    ) -> list[Segment]:
        segments: list[Segment] = []
        step = pd.Timedelta(interval)
        current_labels: list[int] = []
        current_times: list[pd.Timestamp] = []
        previous_time: pd.Timestamp | None = None
        for label, time, finite in zip(rows["label"], rows["time"], rows["finite"], strict=True):
            time = pd.Timestamp(time)
            contiguous = previous_time is not None and (time - previous_time) == step
            if not finite or not contiguous:
                if finite and not contiguous:
                    self._flush(segments, station, partition, current_labels, current_times)
                    current_labels, current_times = [int(label)], [time]
                else:
                    self._flush(segments, station, partition, current_labels, current_times)
                    current_labels, current_times = [], []
            else:
                current_labels.append(int(label))
                current_times.append(time)
            previous_time = time
        self._flush(segments, station, partition, current_labels, current_times)
        return segments

    @staticmethod
    def _flush(
        segments: list[Segment],
        station: str,
        partition: str,
        labels: list[int],
        times: list[pd.Timestamp],
    ) -> None:
        if labels:
            segments.append(
                Segment(
                    station=station,
                    partition=partition,
                    row_labels=tuple(labels),
                    times=tuple(time.to_pydatetime() for time in times),
                )
            )
