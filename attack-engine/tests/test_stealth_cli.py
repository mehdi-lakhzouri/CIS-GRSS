from __future__ import annotations

import json
from pathlib import Path

import yaml

from attack_engine.cli import EXIT_INPUT_ERROR, EXIT_OK, main
from attack_engine.utils.hashing import sha256_file


def _arguments(tmp_path, stealth_frame, partition_plan, stealth_parameters):
    data = tmp_path / "clean.parquet"
    stealth_frame.to_parquet(data, index=False)
    config = tmp_path / "dataset.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "validation_profile": {"name": "fixture", "allow_temporal_gaps": True},
                "partitions": partition_plan.model_dump(mode="json")["partitions"],
            }
        )
    )
    params = tmp_path / "attacks.yaml"
    params.write_text(
        yaml.safe_dump(
            {
                "global_seed": 14,
                "quantum": 0.001,
                "stealth": stealth_parameters.model_dump(mode="json"),
            }
        )
    )
    schedule = tmp_path / "scheduling.yaml"
    schedule.write_text(
        yaml.safe_dump(
            {
                "global_seed": 14,
                "attack_budget": 0.2,
                "budget_scope": ["validation", "test"],
                "enabled_families": ["stealth"],
                "family_parameters": {"stealth": {"weight": 1, "min_history_hours": 4}},
                "min_window_length": 6,
                "max_window_length": 12,
                "max_attempts": 100,
            }
        )
    )
    return ["--input", str(data), "--config", str(config), "--attack-params", str(params)], schedule


def test_calibrate_writes_only_its_artifact_then_generate_is_read_only(
    tmp_path,
    stealth_frame,
    partition_plan,
    stealth_parameters,
    capsys,
):
    args, schedule = _arguments(tmp_path, stealth_frame, partition_plan, stealth_parameters)
    data = Path(args[1])
    before_hash = sha256_file(data)
    files_before = set(tmp_path.rglob("*"))
    code = main(
        ["calibrate", *args, "--output", str(stealth_parameters.calibration_path), "--json"]
    )
    assert code == EXIT_OK
    artifact = json.loads(capsys.readouterr().out)["calibration"]
    assert artifact["training_partition"] == "train"
    assert set(tmp_path.rglob("*")) - files_before == {stealth_parameters.calibration_path}
    files_before = set(tmp_path.rglob("*"))
    command = [
        "generate",
        *args,
        "--attack-config",
        str(schedule),
        "--json",
        "--on-unsupported-family",
        "error",
    ]
    assert main(command) == EXIT_OK
    first_output = capsys.readouterr().out
    execution = json.loads(first_output)["execution"]
    assert execution["per_family"]["stealth"]["applied_windows"] > 0
    assert not execution["skipped_families"]
    assert execution["calibration_sha256"] == artifact["calibration_sha256"]
    assert execution["context_coverage"]["fraction"] == 1.0
    assert main(command) == EXIT_OK
    assert capsys.readouterr().out == first_output
    assert set(tmp_path.rglob("*")) == files_before
    assert sha256_file(data) == before_hash


def test_missing_calibration_is_a_clean_cli_error(
    tmp_path,
    stealth_frame,
    partition_plan,
    stealth_parameters,
    capsys,
):
    args, schedule = _arguments(tmp_path, stealth_frame, partition_plan, stealth_parameters)
    code = main(["generate", *args, "--attack-config", str(schedule), "--json"])
    captured = capsys.readouterr()
    assert code == EXIT_INPUT_ERROR
    assert "cannot load calibration" in captured.err
    assert not captured.out


def test_calibration_has_to_be_requested_explicitly_and_never_overwrites_a_dataset(
    tmp_path,
    stealth_frame,
    partition_plan,
    stealth_parameters,
    capsys,
):
    args, _ = _arguments(tmp_path, stealth_frame, partition_plan, stealth_parameters)
    data = Path(args[1])
    before = data.read_bytes()
    assert main(["calibrate", *args, "--output", str(data)]) == EXIT_INPUT_ERROR
    assert "separate .json" in capsys.readouterr().err
    assert data.read_bytes() == before
