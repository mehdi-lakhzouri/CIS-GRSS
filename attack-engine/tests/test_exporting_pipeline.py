from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from attack_engine.attacks.config import load_attack_config
from attack_engine.attacks.executor import AttackExecutor
from attack_engine.dataset.loader import DatasetLoader
from attack_engine.exporting.models import ExportConfiguration, ExportError
from attack_engine.exporting.pipeline import export_run
from attack_engine.scheduling.config import load_scheduling_config
from attack_engine.scheduling.scheduler import AttackWindowScheduler
from attack_engine.utils.hashing import sha256_file

_SCHEDULING_YAML = """
global_seed: 14
attack_budget: 0.3
budget_scope: [validation]
enabled_families: [spike, bias, drift]
min_window_length: 2
max_window_length: 24
max_attempts: 2000
family_parameters:
  spike: {weight: 1.0, min_window: 1, max_window: 3}
  bias: {weight: 1.0, min_window: 6, max_window: 24}
  drift: {weight: 1.0, min_window: 12, max_window: 48}
"""

_ATTACKS_YAML = """
global_seed: 14
quantum: 0.001
spike: {magnitude: 0.2, min_points: 1, max_points: 3}
bias: {magnitude: 0.05}
drift: {final_magnitude: 0.08, max_slope_per_hour: 0.05}
"""

EXPECTED_FILES = (
    "uscrn_attacked.parquet",
    "detector_input.parquet",
    "ground_truth.parquet",
    "ground_truth_cells.parquet",
    "detector",
    "attack_manifest.csv",
    "run_provenance.json",
    "data_contract.json",
    "SHA256SUMS.txt",
    "HANDOVER.md",
)


@pytest.fixture
def exported(schema, partition_plan, make_observations, tmp_path: Path):
    frame = make_observations(hours=336)
    input_path = tmp_path / "input.parquet"
    frame.to_parquet(input_path, index=False)
    dataset = DatasetLoader(schema=schema).load(input_path)

    scheduling_path = tmp_path / "scheduling.yaml"
    scheduling_path.write_text(_SCHEDULING_YAML, encoding="utf-8")
    attacks_path = tmp_path / "attacks.yaml"
    attacks_path.write_text(_ATTACKS_YAML, encoding="utf-8")

    scheduling_config = load_scheduling_config(scheduling_path)
    attack_config = load_attack_config(attacks_path)
    plan = AttackWindowScheduler(schema, partition_plan, scheduling_config).plan(dataset.frame)
    summary = AttackExecutor(schema, partition_plan, attack_config).execute(dataset.frame, plan)

    export_config = ExportConfiguration(output_root=tmp_path / "generated", run_id="run-test")
    kwargs = {
        "dataset": dataset,
        "schema": schema,
        "partitions": partition_plan,
        "scheduling_config_path": scheduling_path,
        "attack_config": attack_config,
        "attack_params_path": attacks_path,
        "plan": plan,
        "summary": summary,
    }
    result = export_run(config=export_config, **kwargs)
    return dataset, plan, summary, result, kwargs


def test_export_writes_every_expected_file(exported) -> None:
    _, _, _, result, _ = exported
    names = {path.name for path in result.run_dir.iterdir()}
    assert names == set(EXPECTED_FILES)


def test_export_row_counts_match_original(exported) -> None:
    dataset, _, _, result, _ = exported
    for name in ("uscrn_attacked.parquet", "detector_input.parquet", "ground_truth.parquet"):
        table = pd.read_parquet(result.run_dir / name)
        assert len(table) == dataset.metadata.row_count
    assert result.row_count == dataset.metadata.row_count


def test_export_reconciles_with_executor(exported) -> None:
    _, _, summary, result, _ = exported
    ground_truth = pd.read_parquet(result.run_dir / "ground_truth.parquet")
    manifest = pd.read_csv(result.run_dir / "attack_manifest.csv")

    assert int(ground_truth["is_attacked"].sum()) == summary.committed_total
    assert (
        int(manifest.loc[manifest["status"] == "applied", "modified_count"].sum())
        == summary.committed_total
    )
    assert result.modified_count == summary.committed_total
    assert result.validation_passed
    assert result.validation_messages == ()


def test_detector_input_has_no_privileged_columns(exported) -> None:
    _, _, _, result, _ = exported
    detector_input = pd.read_parquet(result.run_dir / "detector_input.parquet")
    assert "is_attacked" not in detector_input.columns
    assert "attack_type" not in detector_input.columns
    assert "attack_id" not in detector_input.columns
    assert "flag_incoherence_verticale" not in detector_input.columns
    clean_target_present = any(
        col.endswith("_clean") or col == "clean_target" for col in detector_input.columns
    )
    assert not clean_target_present


def test_missing_targets_remain_missing(exported) -> None:
    dataset, _, _, result, _ = exported
    ground_truth = pd.read_parquet(result.run_dir / "ground_truth.parquet")
    missing_original = dataset.frame["SOIL_MOISTURE_5"].isna()
    assert ground_truth.loc[missing_original.to_numpy(), "attacked_target"].isna().all()
    assert not ground_truth.loc[missing_original.to_numpy(), "is_attacked"].any()
    assert not ground_truth.loc[missing_original.to_numpy(), "is_evaluable"].any()


def test_checksums_file_covers_provenance(exported) -> None:
    _, _, _, result, _ = exported
    lines = (result.run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    recorded = {name: digest for digest, name in (line.split("  ", 1) for line in lines)}
    assert "run_provenance.json" in recorded
    assert sha256_file(result.run_dir / "run_provenance.json") == recorded["run_provenance.json"]
    # The provenance document never hashes itself.
    provenance = json.loads((result.run_dir / "run_provenance.json").read_text(encoding="utf-8"))
    assert "run_provenance.json" not in provenance["output_files"]
    assert "SHA256SUMS.txt" not in provenance["output_files"]


def test_source_file_is_never_modified(exported) -> None:
    dataset, *_ = exported
    assert sha256_file(Path(dataset.metadata.path)) == dataset.metadata.sha256


def test_run_directory_is_not_overwritten(exported) -> None:
    _, _, _, result, kwargs = exported
    repeat_config = ExportConfiguration(
        output_root=result.run_dir.parent, run_id=result.run_dir.name
    )
    with pytest.raises(ExportError, match="already exists"):
        export_run(config=repeat_config, **kwargs)


def test_failed_write_leaves_no_partial_or_temp_directory(
    exported, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, result, kwargs = exported
    output_root = result.run_dir.parent

    def _boom(self, *args, **kwargs):
        raise OSError("simulated disk failure")

    # Patch the underlying pandas write so the real write_csv still wraps it as
    # ExportError, exercising the pipeline's actual error-handling path.
    monkeypatch.setattr(pd.DataFrame, "to_csv", _boom)
    new_config = ExportConfiguration(output_root=output_root, run_id="run-will-fail")

    with pytest.raises(ExportError):
        export_run(config=new_config, **kwargs)

    remaining = {path.name for path in output_root.iterdir()}
    assert "run-will-fail" not in remaining
    assert not any(name.startswith(".tmp-run-will-fail") for name in remaining)
