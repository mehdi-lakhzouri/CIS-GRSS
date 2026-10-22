"""Small deterministic synthetic fixtures shaped like the USCRN hourly table."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attack_engine.attacks.config import (
    AttackExecutionConfig,
    BiasParameters,
    DriftParameters,
    ReplayParameters,
    SpikeParameters,
    StealthParameters,
)
from attack_engine.attacks.models import AttackRequest, ReplayHistory, StealthContext
from attack_engine.dataset.schema import (
    FIXTURE_PROFILE,
    DatasetSchema,
    PartitionPlan,
    PartitionSpec,
    uscrn_hourly_schema,
)
from attack_engine.dataset.validator import DatasetValidator
from attack_engine.scheduling.config import FamilyParameters, SchedulingConfig
from attack_engine.scheduling.models import AttackFamily, AttackWindow

FIXTURE_STATIONS = ("AZ_Alpha_1_N", "OK_Beta_2_SE")

ObservationFactory = Callable[..., pd.DataFrame]
SchedulingConfigFactory = Callable[..., SchedulingConfig]
AttackWindowFactory = Callable[..., AttackWindow]
AttackRequestFactory = Callable[..., AttackRequest]
AttackConfigFactory = Callable[..., AttackExecutionConfig]


def build_observations(
    start: str = "2024-03-01 01:00",
    hours: int = 48,
    stations: Sequence[str] = FIXTURE_STATIONS,
) -> pd.DataFrame:
    """Build a complete, valid hourly table: one row per station and interval end."""
    frames = []
    for index, station in enumerate(stations):
        time = pd.date_range(start, periods=hours, freq="h", unit="us")
        # Daily cycle keeps every value within physical bounds for any fixture length.
        step = np.arange(hours, dtype="float64") % 24
        constant = np.ones(hours)
        frames.append(
            pd.DataFrame(
                {
                    "time": time,
                    "station": station,
                    "annee": (time - pd.Timedelta(hours=1)).year.astype("int64"),
                    "LATITUDE": 30.0 + index,
                    "LONGITUDE": -100.0 - index,
                    "T_HR_AVG": 10.0 + 0.1 * step,
                    "T_MAX": 10.5 + 0.1 * step,
                    "T_MIN": 9.5 + 0.1 * step,
                    "P_CALC": np.where(step % 10 == 0, 0.5, 0.0),
                    "SOLARAD": 100.0 * constant,
                    "SUR_TEMP": 12.0 * constant,
                    "RH_HR_AVG": 50.0 * constant,
                    "SOIL_MOISTURE_5": 0.200 + 0.001 * step,
                    "SOIL_MOISTURE_10": 0.220 * constant,
                    "SOIL_MOISTURE_20": 0.250 * constant,
                    "SOIL_MOISTURE_50": 0.280 * constant,
                    "SOIL_MOISTURE_100": 0.300 * constant,
                    "SOIL_TEMP_5": 15.0 * constant,
                    "SOIL_TEMP_10": 15.5 * constant,
                    "SOIL_TEMP_20": 16.0 * constant,
                    "SOIL_TEMP_50": 16.5 * constant,
                    "SOIL_TEMP_100": 17.0 * constant,
                    "flag_incoherence_verticale": False,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def schema() -> DatasetSchema:
    return uscrn_hourly_schema()


@pytest.fixture
def validator(schema: DatasetSchema) -> DatasetValidator:
    return DatasetValidator(schema=schema, profile=FIXTURE_PROFILE)


@pytest.fixture
def make_observations() -> ObservationFactory:
    return build_observations


@pytest.fixture
def observations() -> pd.DataFrame:
    return build_observations()


@pytest.fixture
def parquet_file(tmp_path: Path, observations: pd.DataFrame) -> Path:
    path = tmp_path / "observations.parquet"
    observations.to_parquet(path, index=False)
    return path


def build_partition_plan() -> PartitionPlan:
    """train 2022-2023, validation 2024, test 2025 - on interval-end UTC bounds."""
    return PartitionPlan(
        partitions=(
            PartitionSpec(name="train", start="2022-01-01T00:00:00Z", end="2024-01-01T00:00:00Z"),
            PartitionSpec(
                name="validation", start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z"
            ),
            PartitionSpec(name="test", start="2025-01-01T00:00:00Z", end="2026-01-01T00:00:00Z"),
        )
    )


@pytest.fixture
def partition_plan() -> PartitionPlan:
    return build_partition_plan()


def build_scheduling_config(
    *,
    global_seed: int = 14,
    attack_budget: float = 0.10,
    budget_scope: Sequence[str] = ("validation",),
    families: Sequence[AttackFamily] = (AttackFamily.SPIKE, AttackFamily.BIAS),
    weights: dict[AttackFamily, float] | None = None,
    min_window_length: int = 2,
    max_window_length: int = 6,
    max_attempts: int = 500,
    min_history: dict[AttackFamily, int] | None = None,
    **overrides: object,
) -> SchedulingConfig:
    weights = weights or dict.fromkeys(families, 1.0)
    min_history = min_history or {}
    family_parameters = {
        family: FamilyParameters(
            weight=weights[family], min_history_hours=min_history.get(family, 0)
        )
        for family in families
    }
    return SchedulingConfig(
        global_seed=global_seed,
        attack_budget=attack_budget,
        budget_scope=tuple(budget_scope),
        enabled_families=tuple(families),
        family_parameters=family_parameters,
        min_window_length=min_window_length,
        max_window_length=max_window_length,
        max_attempts=max_attempts,
        **overrides,
    )


@pytest.fixture
def make_scheduling_config() -> SchedulingConfigFactory:
    return build_scheduling_config


def build_attack_window(
    family: AttackFamily,
    *,
    station: str = "AZ_Alpha_1_N",
    partition: str = "validation",
    start: str = "2024-03-01 01:00",
    length: int = 4,
    labels: Sequence[int] | None = None,
    sequence: int = 0,
) -> AttackWindow:
    times = pd.date_range(start, periods=length, freq="h", tz="UTC")
    row_labels = tuple(labels) if labels is not None else tuple(range(length))
    return AttackWindow(
        window_id=f"{family.value}-{station}-{partition}-{sequence:04d}",
        family=family,
        station=station,
        partition=partition,
        length=length,
        first_observation_time=times[0].to_pydatetime(),
        last_observation_time=times[-1].to_pydatetime(),
        end_exclusive=(times[-1] + pd.Timedelta(hours=1)).to_pydatetime(),
        row_labels=row_labels,
    )


def build_attack_request(
    family: AttackFamily,
    clean_values: Sequence[float],
    *,
    window: AttackWindow | None = None,
    history: ReplayHistory | None = None,
    interval_hours: float = 1.0,
    stealth_context: StealthContext | None = None,
    **window_kwargs: object,
) -> AttackRequest:
    clean = tuple(float(value) for value in clean_values)
    window = window or build_attack_window(family, length=len(clean), **window_kwargs)
    elapsed = tuple(i * interval_hours for i in range(len(clean)))
    return AttackRequest(
        window=window,
        clean_values=clean,
        elapsed_hours=elapsed,
        history=history,
        stealth_context=stealth_context,
    )


def build_attack_config(
    *,
    global_seed: int = 14,
    quantum: float | None = None,
    spike: dict | None = None,
    bias: dict | None = None,
    drift: dict | None = None,
    replay: dict | None = None,
    stealth: dict | None = None,
    **overrides: object,
) -> AttackExecutionConfig:
    return AttackExecutionConfig(
        global_seed=global_seed,
        quantum=quantum,
        spike=SpikeParameters(**spike) if spike is not None else None,
        bias=BiasParameters(**bias) if bias is not None else None,
        drift=DriftParameters(**drift) if drift is not None else None,
        replay=ReplayParameters(**replay) if replay is not None else None,
        stealth=StealthParameters(**stealth) if stealth is not None else None,
        **overrides,
    )


@pytest.fixture
def make_attack_window() -> AttackWindowFactory:
    return build_attack_window


@pytest.fixture
def make_attack_request() -> AttackRequestFactory:
    return build_attack_request


@pytest.fixture
def make_attack_config() -> AttackConfigFactory:
    return build_attack_config


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def stealth_parameters(tmp_path: Path) -> StealthParameters:
    return StealthParameters(
        calibration_path=tmp_path / "calibration.json",
        min_history_hours=4,
        min_training_rows=24,
        residual_floor=0.03,
        magnitude_min=0.01,
        magnitude_max=0.01,
        sign="positive",
        max_change_rate=0.003,
        min_attack_impact=0.002,
    )


@pytest.fixture
def stealth_frame(make_observations: ObservationFactory) -> pd.DataFrame:
    return pd.concat(
        [
            make_observations(start="2023-06-01 01:00", hours=240),
            make_observations(start="2024-03-01 01:00", hours=96),
            make_observations(start="2025-03-01 01:00", hours=96),
        ],
        ignore_index=True,
    )


@pytest.fixture
def stealth_calibration(stealth_frame, schema, partition_plan, stealth_parameters):
    from attack_engine.attacks.calibration import fit_calibration, save_calibration

    artifact = fit_calibration(stealth_frame, schema, partition_plan, stealth_parameters)
    save_calibration(artifact, stealth_parameters.calibration_path)
    return artifact


@pytest.fixture
def make_stealth_request(stealth_frame, stealth_calibration, make_attack_window):
    def build(length=12, offset=24, station=FIXTURE_STATIONS[0], partition="validation"):
        year = {"train": 2023, "validation": 2024, "test": 2025}[partition]
        group = stealth_frame.loc[
            stealth_frame.station.eq(station) & stealth_frame.time.dt.year.eq(year)
        ]
        rows = group.iloc[offset : offset + length]
        times = tuple(ts.to_pydatetime() for ts in rows.time.dt.tz_localize("UTC"))
        window = make_attack_window(
            AttackFamily.STEALTH,
            station=station,
            partition=partition,
            start=times[0].isoformat(),
            length=length,
            labels=rows.index,
        )
        n = stealth_calibration.history_samples
        prefix = group.iloc[max(0, offset - n) : offset]
        context = StealthContext(
            times=times,
            measurements={name: tuple(rows[name]) for name in stealth_calibration.context_features},
            history_times=tuple(ts.to_pydatetime() for ts in prefix.time.dt.tz_localize("UTC")),
            history_values=tuple(prefix.SOIL_MOISTURE_5),
            calibration=stealth_calibration,
        )
        return AttackRequest(
            window=window,
            clean_values=tuple(rows.SOIL_MOISTURE_5),
            elapsed_hours=tuple(float(i) for i in range(length)),
            stealth_context=context,
        )

    return build
