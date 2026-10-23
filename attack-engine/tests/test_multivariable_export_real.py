"""Real USCRN export integrations: all independent campaigns and the combined run."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from attack_engine.attacks.config import load_attack_config
from attack_engine.attacks.executor import AttackExecutor
from attack_engine.config import load_config
from attack_engine.dataset.loader import DatasetLoader
from attack_engine.dataset.schema import uscrn_hourly_schema
from attack_engine.exporting.models import ExportConfiguration
from attack_engine.exporting.pipeline import export_run
from attack_engine.scheduling.config import load_scheduling_config
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler
from attack_engine.utils.hashing import sha256_file
from test_multivariable_export import DATA_FILES, assert_export_matches_execution

PROJECT = Path(__file__).resolve().parents[1]
REAL = os.environ.get("ATTACK_ENGINE_REAL_DATASET")
pytestmark = [
    pytest.mark.real_dataset,
    pytest.mark.skipif(not REAL, reason="ATTACK_ENGINE_REAL_DATASET is not set"),
]


@pytest.fixture(scope="module")
def real_export_dataset():
    schema = uscrn_hourly_schema()
    config = load_config(PROJECT / "configs/dataset.yaml")
    dataset = DatasetLoader(schema).load(Path(str(REAL)))
    assert dataset.source_sha256 == config.expected_sha256
    return schema, config, dataset


@pytest.mark.parametrize("campaign", [*(f.value for f in AttackFamily), "combined"])
def test_real_campaign_export_is_exact_reproducible_and_restricted(
    campaign,
    real_export_dataset,
    tmp_path,
):
    schema, dataset_config, dataset = real_export_dataset
    root = PROJECT / "configs/campaigns" / campaign
    schedule_path, params_path = root / "scheduling.yaml", root / "attacks.yaml"
    scheduling, params = load_scheduling_config(schedule_path), load_attack_config(params_path)
    scheduler = AttackWindowScheduler(schema, dataset_config.partitions, scheduling)
    executor = AttackExecutor(schema, dataset_config.partitions, params)
    snapshot = dataset.frame.copy(deep=True)
    kwargs = {
        "dataset": dataset,
        "schema": schema,
        "partitions": dataset_config.partitions,
        "scheduling_config_path": schedule_path,
        "attack_config": params,
        "attack_params_path": params_path,
    }
    first_plan = scheduler.plan(dataset.frame)
    first_summary = executor.execute(dataset.frame, first_plan)
    assert all(p.applied_windows for p in first_summary.per_family.values())
    first_kwargs = {**kwargs, "plan": first_plan, "summary": first_summary}
    first = export_run(
        config=ExportConfiguration(output_root=tmp_path / "runs", run_id="a"), **first_kwargs
    )
    assert_export_matches_execution(first, first_kwargs)
    second_plan = scheduler.plan(dataset.frame)
    second_summary = executor.execute(dataset.frame, second_plan)
    second = export_run(
        config=ExportConfiguration(output_root=tmp_path / "runs", run_id="b"),
        **kwargs,
        plan=second_plan,
        summary=second_summary,
    )
    assert {name: first.file_sha256[name] for name in DATA_FILES} == {
        name: second.file_sha256[name] for name in DATA_FILES
    }
    pd.testing.assert_frame_equal(dataset.frame, snapshot, check_exact=True)
    assert sha256_file(Path(dataset.metadata.path)) == dataset.source_sha256
