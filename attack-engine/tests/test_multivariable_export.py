"""Full multi-variable exports, audience separation and fail-safe read-back validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from attack_engine.attacks.calibration import fit_calibration, save_calibration
from attack_engine.attacks.config import load_attack_config
from attack_engine.attacks.executor import AttackExecutor
from attack_engine.dataset.loader import DatasetLoader
from attack_engine.exporting import pipeline
from attack_engine.exporting.models import ExportConfiguration, ExportError
from attack_engine.exporting.pipeline import export_run
from attack_engine.exporting.writer import run_in_temp_directory
from attack_engine.scheduling.config import load_scheduling_config
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler
from attack_engine.utils.hashing import sha256_file

TARGETS = ("SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC")
DATA_FILES = (
    "uscrn_attacked.parquet",
    "detector_input.parquet",
    "ground_truth.parquet",
    "ground_truth_cells.parquet",
    "attack_manifest.csv",
    "data_contract.json",
    "detector/detector_input.parquet",
    "detector/row_labels.parquet",
    "detector/cell_labels.parquet",
    "detector/data_contract.json",
    "detector/HANDOVER.md",
    "detector/SHA256SUMS.txt",
)


@pytest.fixture
def prepared_export(tmp_path, schema, partition_plan, stealth_frame, stealth_parameters):
    frame = stealth_frame.copy(deep=True)
    frame[TARGETS[1]] = (0.8 * frame[TARGETS[0]] + 0.05).round(3)
    frame.loc[2, TARGETS[0]] = np.nan
    frame.loc[260, TARGETS[1]] = np.nan
    save_calibration(
        fit_calibration(frame, schema, partition_plan, stealth_parameters),
        stealth_parameters.calibration_path,
    )
    path = tmp_path / "clean.parquet"
    frame.to_parquet(path, index=False)
    dataset = DatasetLoader(schema).load(path)

    def prepare(families=(AttackFamily.BIAS,), targets=TARGETS, budget_unit="cells", budget=0.9):
        magnitudes = dict(zip(TARGETS, (0.02, 0.01, 0.5), strict=True))
        common = {"target_columns": list(targets)}
        additive = {
            **common,
            "sign": "positive",
            "secondary_magnitudes": {c: magnitudes[c] for c in targets[1:]},
        }
        all_params = {
            "spike": {
                **additive,
                "magnitude": magnitudes[targets[0]],
                "min_points": 2,
                "max_points": 3,
            },
            "bias": {**additive, "magnitude": magnitudes[targets[0]]},
            "drift": {**additive, "final_magnitude": magnitudes[targets[0]]},
            "replay": {**common, "min_age_hours": 1},
            "stealth": {
                **stealth_parameters.model_dump(mode="json"),
                **common,
                "secondary_columns": [c for c in targets if c != TARGETS[0]],
                "rainfall_lag_hours": 0,
                "rainfall_trigger": 0.001,
            },
        }
        scheduling_path = tmp_path / "scheduling.yaml"
        attacks_path = tmp_path / "attacks.yaml"
        scheduling_path.write_text(
            yaml.safe_dump(
                {
                    "global_seed": 14,
                    "attack_budget": budget,
                    "budget_scope": ["validation", "test"],
                    "enabled_families": [f.value for f in families],
                    "min_window_length": 3,
                    "max_window_length": 12,
                    "max_attempts": 1000,
                    "budget_unit": budget_unit,
                    "eligibility_target": targets[0],
                    "family_parameters": {
                        f.value: {
                            "weight": 1,
                            "variable_count": len(targets),
                            "min_history_hours": 12,
                        }
                        for f in families
                    },
                }
            ),
            encoding="utf-8",
        )
        attacks_path.write_text(
            yaml.safe_dump(
                {
                    "global_seed": 14,
                    "quantum": 0.001,
                    **{f.value: all_params[f.value] for f in families},
                }
            ),
            encoding="utf-8",
        )
        scheduling = load_scheduling_config(scheduling_path)
        config = load_attack_config(attacks_path)
        scheduler = AttackWindowScheduler(schema, partition_plan, scheduling)
        plan = scheduler.plan(dataset.frame)
        summary = AttackExecutor(schema, partition_plan, config).execute(dataset.frame, plan)
        return {
            "dataset": dataset,
            "schema": schema,
            "partitions": partition_plan,
            "scheduling_config_path": scheduling_path,
            "attack_config": config,
            "attack_params_path": attacks_path,
            "plan": plan,
            "summary": summary,
        }

    return prepare


def assert_export_matches_execution(result, kwargs):
    source = kwargs["dataset"].frame
    summary = kwargs["summary"]
    observed = pd.read_parquet(result.run_dir / "detector_input.parquet")
    rows = pd.read_parquet(result.run_dir / "ground_truth.parquet")
    cells = pd.read_parquet(result.run_dir / "ground_truth_cells.parquet")
    manifest = pd.read_csv(result.run_dir / "attack_manifest.csv")
    contract = json.loads((result.run_dir / "data_contract.json").read_text(encoding="utf-8"))
    provenance = json.loads((result.run_dir / "run_provenance.json").read_text(encoding="utf-8"))
    assert tuple(observed.columns) == tuple(contract["detector_input_columns"])
    assert not {
        "is_attacked",
        "attack_id",
        "attack_type",
        "flag_incoherence_verticale",
        "clean_target",
    } & set(observed)
    assert not any(c.endswith(("_clean", "_attacked")) for c in observed)
    assert len(cells) == len(source) * len(result.target_columns)
    assert not cells.duplicated(["station", "time", "column"]).any()
    assert int(rows.is_attacked.sum()) == result.modified_rows == summary.committed_rows
    assert int(cells.is_attacked.sum()) == result.modified_cells == summary.committed_cells
    assert provenance["modified_rows"] == result.modified_rows
    assert provenance["modified_cells"] == result.modified_cells
    assert provenance["budget_unit"] == kwargs["plan"].budget_unit
    assert provenance["train_preserved"]
    assert result.validation_passed
    assert result.modified_count == result.modified_rows
    expected = source.loc[:, result.target_columns].copy()
    events = {}
    for event in summary.applied_results:
        for column in event.columns:
            for label, value, changed in zip(
                column.row_labels, column.candidate_values, column.changed_mask, strict=True
            ):
                if changed:
                    assert (label, column.column) not in events
                    events[(label, column.column)] = (event.family.value, event.window_id)
                    expected.at[label, column.column] = value
    for name in result.target_columns:
        table = cells.loc[cells.column.eq(name)].reset_index(drop=True)
        clean = source[name].to_numpy()
        final = observed[name].to_numpy()
        actual = np.isfinite(clean) & np.isfinite(final) & (clean != final)
        np.testing.assert_array_equal(final, expected[name])
        np.testing.assert_array_equal(table.clean_value, clean)
        np.testing.assert_array_equal(table.attacked_value, final)
        np.testing.assert_array_equal(table.is_attacked, actual)
        np.testing.assert_array_equal(table.is_evaluable, np.isfinite(clean))
        np.testing.assert_array_equal(np.isnan(final), np.isnan(clean))
        assert int(actual.sum()) == result.per_variable_modifications.get(name, 0)
        for pos in np.flatnonzero(actual):
            assert (table.attack_type.iloc[pos], table.attack_id.iloc[pos]) == events[
                (source.index[pos], name)
            ]
    for partition in ("train",):
        mask = observed.partition.eq(partition)
        for name in result.target_columns:
            np.testing.assert_array_equal(observed.loc[mask, name], source.loc[mask, name])
    for name in ("row_labels.parquet", "cell_labels.parquet"):
        labels = pd.read_parquet(result.detector_release_dir / name)
        assert set(labels.partition) <= {"train", "validation"}
        assert not any(c.startswith(("clean", "attacked")) for c in labels)
    assert {p.name for p in result.detector_release_dir.iterdir()} == {
        "detector_input.parquet",
        "row_labels.parquet",
        "cell_labels.parquet",
        "data_contract.json",
        "HANDOVER.md",
        "SHA256SUMS.txt",
    }
    assert (
        sha256_file(result.detector_release_dir / "detector_input.parquet")
        == result.file_sha256["detector_input.parquet"]
    )
    for directory in (result.run_dir, result.detector_release_dir):
        for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            assert sha256_file(directory / name) == digest
    assert sha256_file(Path(kwargs["dataset"].metadata.path)) == kwargs["dataset"].metadata.sha256
    if "modified_cells" in manifest:
        assert int(manifest.modified_cells.sum()) == result.modified_cells
        assert int(manifest.modified_rows.sum()) == result.modified_rows
        assert int(manifest.committed_budget.sum()) == summary.committed_total
    return rows, cells, manifest


@pytest.mark.parametrize("family", tuple(AttackFamily))
@pytest.mark.parametrize("unit", ("cells", "rows"))
def test_all_families_export_every_variable_and_exact_cell_truth(
    family, unit, prepared_export, tmp_path
):
    kwargs = prepared_export(families=(family,), budget_unit=unit)
    assert kwargs["summary"].applied_results
    result = export_run(
        config=ExportConfiguration(output_root=tmp_path / "runs", run_id="first"), **kwargs
    )
    assert_export_matches_execution(result, kwargs)
    assert result.target_columns == TARGETS
    assert result.cell_ground_truth_row_count == len(kwargs["dataset"].frame) * 3
    if family is AttackFamily.STEALTH:
        provenance = json.loads(
            (result.run_dir / "run_provenance.json").read_text(encoding="utf-8")
        )
        assert provenance["secondary_calibration"]["partition_end"].startswith("2024-01-01")
        assert provenance["secondary_calibration_sha256"]


@pytest.mark.parametrize("targets", [(TARGETS[1],), (TARGETS[2],), (TARGETS[2], TARGETS[0])])
def test_export_supports_nondefault_primary_and_subsets(targets, prepared_export, tmp_path):
    kwargs = prepared_export(targets=targets)
    result = export_run(config=ExportConfiguration(output_root=tmp_path / "runs"), **kwargs)
    assert_export_matches_execution(result, kwargs)
    assert result.target_columns == targets


def test_combined_export_and_reproduction_are_deterministic(prepared_export, tmp_path):
    kwargs = prepared_export(families=tuple(AttackFamily))
    assert {r.family for r in kwargs["summary"].applied_results} == set(AttackFamily)
    first = export_run(
        config=ExportConfiguration(output_root=tmp_path / "runs", run_id="first"), **kwargs
    )
    assert_export_matches_execution(first, kwargs)
    second_kwargs = prepared_export(families=tuple(AttackFamily))
    second = export_run(
        config=ExportConfiguration(output_root=tmp_path / "runs", run_id="second"), **second_kwargs
    )
    assert_export_matches_execution(second, second_kwargs)
    assert {name: first.file_sha256[name] for name in DATA_FILES} == {
        name: second.file_sha256[name] for name in DATA_FILES
    }


@pytest.mark.parametrize(
    "partitions", [("test",), ("train", "test"), ("unknown",), ("train", "train")]
)
def test_detector_label_policy_cannot_release_test_truth(partitions):
    with pytest.raises(ValidationError, match="train/validation"):
        ExportConfiguration(output_root="unused", detector_label_partitions=partitions)


@pytest.mark.parametrize("run_id", ["../prior", "..", "sub/run", "sub\\run", "C:/prior"])
def test_run_id_cannot_escape_output_root(run_id):
    with pytest.raises(ValidationError):
        ExportConfiguration(output_root="unused", run_id=run_id)


@pytest.mark.parametrize("artifact", ["detector_input.parquet", "ground_truth_cells.parquet"])
def test_written_corruption_fails_before_publish(artifact, prepared_export, tmp_path, monkeypatch):
    kwargs = prepared_export()
    original = pipeline.write_parquet

    def corrupt(path, frame):
        frame = frame.copy(deep=True)
        if path.name == artifact:
            if artifact == "ground_truth_cells.parquet":
                row = frame.index[frame.is_attacked][0]
                frame.loc[row, "is_attacked"] = False
            else:
                frame[TARGETS[1]] = kwargs["dataset"].frame[TARGETS[1]].to_numpy()
        original(path, frame)

    monkeypatch.setattr(pipeline, "write_parquet", corrupt)
    root = tmp_path / "runs"
    with pytest.raises(ExportError, match="consistency"):
        export_run(config=ExportConfiguration(output_root=root, run_id="bad"), **kwargs)
    assert list(root.iterdir()) == []


def test_existing_export_is_never_overwritten(prepared_export, tmp_path):
    kwargs = prepared_export()
    config = ExportConfiguration(output_root=tmp_path / "runs", run_id="existing")
    first = export_run(config=config, **kwargs)
    original_hashes = {
        p.relative_to(first.run_dir).as_posix(): sha256_file(p)
        for p in first.run_dir.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ExportError, match="already exists"):
        export_run(config=config, **kwargs)
    assert original_hashes == {
        p.relative_to(first.run_dir).as_posix(): sha256_file(p)
        for p in first.run_dir.rglob("*")
        if p.is_file()
    }


def test_publication_race_preserves_existing_run_and_cleans_staging(tmp_path):
    def write(temp):
        (temp / "new.txt").write_text("new", encoding="utf-8")
        prior = tmp_path / "run"
        prior.mkdir()
        (prior / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ExportError, match="already exists"):
        run_in_temp_directory(tmp_path, "run", write)
    assert (tmp_path / "run/old.txt").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".tmp-*"))


def test_zero_budget_exports_full_negative_cell_truth(prepared_export, tmp_path):
    kwargs = prepared_export(budget=0)
    result = export_run(config=ExportConfiguration(output_root=tmp_path / "zero"), **kwargs)
    assert_export_matches_execution(result, kwargs)
    assert result.modified_cells == result.modified_rows == 0
    assert result.per_variable_modifications == dict.fromkeys(TARGETS, 0)
    assert result.manifest_row_count == 0


def test_export_rejects_mutated_in_memory_reference(prepared_export, tmp_path):
    kwargs = prepared_export()
    kwargs["dataset"].frame.loc[0, "T_HR_AVG"] += 1
    with pytest.raises(ExportError, match="clean frame differs"):
        export_run(config=ExportConfiguration(output_root=tmp_path / "bad"), **kwargs)
    assert not (tmp_path / "bad").exists()


def test_detector_release_can_withhold_all_training_labels(prepared_export, tmp_path):
    kwargs = prepared_export()
    result = export_run(
        config=ExportConfiguration(
            output_root=tmp_path / "unlabelled",
            detector_label_partitions=(),
        ),
        **kwargs,
    )
    for name in ("row_labels.parquet", "cell_labels.parquet"):
        assert pd.read_parquet(result.detector_release_dir / name).empty


def test_row_truth_aggregates_disjoint_cell_events(make_observations, schema, partition_plan):
    from attack_engine.attacks.base import build_result
    from attack_engine.attacks.executor import ExecutionSummary
    from attack_engine.attacks.models import (
        AttackRequest,
    )
    from attack_engine.exporting.builder import (
        assign_row_partitions,
        build_attacked_columns,
        build_multivariable_ground_truth,
    )
    from conftest import build_attack_window

    frame = make_observations(hours=4)
    results = []
    for name, family, event_id in (
        (TARGETS[0], AttackFamily.SPIKE, "a"),
        (TARGETS[1], AttackFamily.BIAS, "b"),
    ):
        window = build_attack_window(family, length=1, labels=(0,)).model_copy(
            update={"window_id": event_id}
        )
        clean = np.array([frame.at[0, name]])
        request = AttackRequest(window, tuple(clean), (0.0,), target_columns=(name,))
        results.append(
            build_result(
                request,
                family,
                clean,
                clean + 0.01,
                np.array([True]),
                np.array([False]),
                requested_magnitude=0.01,
                sign=1,
            )
        )
    summary = ExecutionSummary(
        14, 2, 2, 2, 0, {}, tuple(results), True, committed_cells=2, committed_rows=1
    )
    overlays = build_attacked_columns(frame, schema, summary, TARGETS[:2])
    truth = build_multivariable_ground_truth(
        frame, schema, overlays, assign_row_partitions(frame, schema, partition_plan)
    )
    assert truth.loc[0, "attack_ids"] == ["a", "b"]
    assert truth.loc[0, "attack_types"] == ["bias", "spike"]
    assert truth.loc[0, "modified_cells"] == 2
    assert truth.loc[0, "changed_columns"] == list(TARGETS[:2])
    assert pd.isna(truth.loc[0, "attack_id"])
    assert truth.loc[0, "attack_type"] == "multiple"
