"""Read-only contextual requests built from the existing finite-segment eligibility engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from attack_engine.attacks.calibration import ContextCalibration
from attack_engine.attacks.models import MultiStealthContext, StealthContext
from attack_engine.attacks.multistealth import SecondaryCalibration
from attack_engine.dataset import temporal
from attack_engine.dataset.schema import DatasetSchema, PartitionPlan
from attack_engine.scheduling.eligibility import EligibilityEngine, Segment
from attack_engine.scheduling.models import AttackFamily, AttackWindow
from attack_engine.scheduling.scheduler import ScheduleResult


class StealthContextProvider:
    """Keep only locations needed by planned windows, not a full history per row."""

    def __init__(
        self,
        frame: pd.DataFrame,
        schema: DatasetSchema,
        partitions: PartitionPlan,
        plan: ScheduleResult,
        calibration: ContextCalibration,
        family: AttackFamily = AttackFamily.STEALTH,
    ) -> None:
        self.frame = frame
        self.schema = schema
        self.calibration = calibration
        self.times = temporal.to_utc(frame[schema.timestamps.column])
        self.locations: dict[int, tuple[Segment, int]] = {}
        first_labels = {w.row_labels[0] for w in plan.windows_for(family)}
        report = EligibilityEngine(schema, partitions, plan.scope).analyze(frame)
        for cell in report.cells:
            for segment in cell.segments:
                for offset, label in enumerate(segment.row_labels):
                    if label in first_labels:
                        self.locations[label] = (segment, offset)

    def for_window(self, window: AttackWindow, emitted: dict[int, float]) -> StealthContext:
        labels = list(window.row_labels)
        measurements = {
            name: tuple(self.frame.loc[labels, name].to_numpy(dtype=float))
            if name in self.frame
            else tuple(np.full(window.length, np.nan))
            for name in self.calibration.context_features
        }
        history_times = ()
        history_values = ()
        location = self.locations.get(window.row_labels[0])
        if location is not None:
            segment, start = location
            lookback = max(0, start - self.calibration.history_samples)
            history_labels = segment.row_labels[lookback:start]
            history_times = segment.times[lookback:start]
            history_values = tuple(
                emitted[label]
                if label in emitted
                else float(self.frame.at[label, self.schema.target_column])
                for label in history_labels
            )
        return StealthContext(
            times=tuple(time.to_pydatetime() for time in self.times.loc[labels]),
            measurements=measurements,
            history_times=history_times,
            history_values=history_values,
            calibration=self.calibration,
        )


class MultiStealthContextProvider:
    """Secondary-variable clean series and train-fitted couplings for coordinated stealth.

    The primary soil-moisture context comes from :class:`StealthContextProvider`; this supplies,
    per secondary column, the clean window values (so missing values are preserved and never
    fabricated) and the frozen secondary calibration.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        schema: DatasetSchema,
        partitions: PartitionPlan,
        plan: ScheduleResult,
        calibration: ContextCalibration,
        secondary_columns: tuple[str, ...],
        secondary_calibration: SecondaryCalibration,
    ) -> None:
        self.frame = frame
        self.secondary_columns = secondary_columns
        self.secondary_calibration = secondary_calibration

    def secondary_context(self, window: AttackWindow) -> MultiStealthContext:
        labels = list(window.row_labels)
        secondary_clean = {
            name: tuple(self.frame.loc[labels, name].to_numpy(dtype=float))
            if name in self.frame
            else tuple(np.full(window.length, np.nan))
            for name in self.secondary_columns
        }
        return MultiStealthContext(
            secondary_clean=secondary_clean, calibration=self.secondary_calibration
        )
