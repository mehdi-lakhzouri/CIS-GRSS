from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from attack_engine.cli import EXIT_INPUT_ERROR, EXIT_OK, EXIT_VALIDATION_FAILED, main
from attack_engine.scheduling.models import AttackFamily
from attack_engine.utils.hashing import sha256_file

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dataset.yaml"
WRONG_HASH = "0" * 64


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every CLI test from an empty directory so no default config is picked up."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return workdir


def _files(root: Path) -> set[Path]:
    return set(root.rglob("*"))


def test_validate_valid_fixture_succeeds(
    parquet_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _files(tmp_path)
    digest = sha256_file(parquet_file)

    code = main(["validate", "--input", str(parquet_file)])

    assert code == EXIT_OK
    output = capsys.readouterr().out
    assert "PASSED" in output
    assert "test-fixture" in output
    assert _files(tmp_path) == before, "the CLI must not write files"
    assert sha256_file(parquet_file) == digest


def test_validate_json_output(parquet_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["validate", "--input", str(parquet_file), "--json"])

    document = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert document["validation"]["profile_name"] == "test-fixture"
    assert document["integrity"]["status"] == "not_configured"
    assert document["dataset"]["sha256"] == sha256_file(parquet_file)


def test_validation_failure_exit_code(
    tmp_path: Path, observations: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "duplicated.parquet"
    pd.concat([observations, observations.iloc[[0]]]).to_parquet(path, index=False)

    code = main(["validate", "--input", str(path)])

    assert code == EXIT_VALIDATION_FAILED
    assert "duplicate_observation_key" in capsys.readouterr().out


def test_hash_mismatch_exit_code(parquet_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["validate", "--input", str(parquet_file), "--expected-sha256", WRONG_HASH])

    assert code == EXIT_VALIDATION_FAILED
    assert "MISMATCH" in capsys.readouterr().out


def test_matching_hash_override(parquet_file: Path) -> None:
    digest = sha256_file(parquet_file)
    assert main(["validate", "--input", str(parquet_file), "--expected-sha256", digest]) == 0


def test_real_profile_rejects_fixture_data(
    parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["validate", "--input", str(parquet_file), "--config", str(PROJECT_CONFIG)])

    output = capsys.readouterr().out
    assert code == EXIT_VALIDATION_FAILED
    assert "row_count_mismatch" in output
    assert "MISMATCH" in output


@pytest.mark.parametrize("command", ["validate", "profile"])
def test_missing_input_exit_code(
    tmp_path: Path, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([command, "--input", str(tmp_path / "absent.parquet")])

    assert code == EXIT_INPUT_ERROR
    assert "not found" in capsys.readouterr().err


def test_unsupported_and_corrupted_inputs_exit_code(tmp_path: Path) -> None:
    text = tmp_path / "data.txt"
    text.write_text("x")
    corrupted = tmp_path / "data.parquet"
    corrupted.write_bytes(b"garbage")

    assert main(["validate", "--input", str(text)]) == EXIT_INPUT_ERROR
    assert main(["validate", "--input", str(corrupted)]) == EXIT_INPUT_ERROR


def test_real_profile_without_config_exit_code(
    parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["validate", "--input", str(parquet_file), "--validation-profile", "real"])

    assert code == EXIT_INPUT_ERROR
    assert "requires a configuration" in capsys.readouterr().err


def test_missing_config_exit_code(parquet_file: Path, tmp_path: Path) -> None:
    args = ["validate", "--input", str(parquet_file), "--config", str(tmp_path / "none.yaml")]
    assert main(args) == EXIT_INPUT_ERROR


def test_malformed_expected_hash_exit_code(parquet_file: Path) -> None:
    args = ["validate", "--input", str(parquet_file), "--expected-sha256", "abc"]
    assert main(args) == EXIT_INPUT_ERROR


def test_usage_errors_exit_with_code_2() -> None:
    with pytest.raises(SystemExit) as no_command:
        main([])
    with pytest.raises(SystemExit) as no_input:
        main(["validate"])
    assert no_command.value.code == 2
    assert no_input.value.code == 2


def test_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["validate", "--help"])
    assert exit_info.value.code == 0
    assert "--input" in capsys.readouterr().out


def test_profile_command(parquet_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["profile", "--input", str(parquet_file), "--json"])

    document = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert document["profile"]["row_count"] == 96
    assert document["profile"]["validation"]["passed"] is True


def test_profile_text_output(parquet_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profile", "--input", str(parquet_file)]) == EXIT_OK
    assert "SOIL_MOISTURE_5" in capsys.readouterr().out


def test_profile_without_station_column_fails(
    tmp_path: Path, observations: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "nostation.parquet"
    observations.drop(columns=["station"]).to_parquet(path, index=False)

    assert main(["profile", "--input", str(path)]) == EXIT_VALIDATION_FAILED
    assert "cannot profile" in capsys.readouterr().err


# plan command ----------------------------------------------------------------------------

_DATASET_CONFIG = """
validation_profile:
  name: fixture-plan
partitions:
  - {name: train, start: '2022-01-01T00:00:00Z', end: '2024-01-01T00:00:00Z'}
  - {name: validation, start: '2024-01-01T00:00:00Z', end: '2025-01-01T00:00:00Z'}
  - {name: test, start: '2025-01-01T00:00:00Z', end: '2026-01-01T00:00:00Z'}
"""

_SCHEDULING_CONFIG = """
global_seed: 7
attack_budget: 0.2
budget_scope: [validation]
enabled_families: [spike, bias]
min_window_length: 2
max_window_length: 6
max_attempts: 500
family_parameters:
  spike: {weight: 1.0}
  bias: {weight: 1.0}
"""


@pytest.fixture
def plan_configs(tmp_path: Path) -> tuple[Path, Path]:
    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(_DATASET_CONFIG, encoding="utf-8")
    scheduling_config = tmp_path / "scheduling.yaml"
    scheduling_config.write_text(_SCHEDULING_CONFIG, encoding="utf-8")
    return dataset_config, scheduling_config


def _plan_args(parquet_file: Path, plan_configs: tuple[Path, Path]) -> list[str]:
    dataset_config, scheduling_config = plan_configs
    return [
        "plan",
        "--input",
        str(parquet_file),
        "--config",
        str(dataset_config),
        "--attack-config",
        str(scheduling_config),
    ]


def test_plan_text_output(
    parquet_file: Path,
    plan_configs: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = _files(tmp_path)
    code = main(_plan_args(parquet_file, plan_configs))

    assert code == EXIT_OK
    output = capsys.readouterr().out
    assert "Seed    : 7" in output
    assert "No observation values were modified" in output
    assert _files(tmp_path) == before, "plan must not write files"


def test_plan_json_output(
    parquet_file: Path, plan_configs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([*_plan_args(parquet_file, plan_configs), "--json"])

    plan = json.loads(capsys.readouterr().out)["plan"]
    assert code == EXIT_OK
    assert plan["seed"] == 7
    assert plan["scope"] == ["validation"]
    assert plan["committed_total"] == 0
    assert plan["reserved_total"] <= plan["requested_quota"]
    assert plan["window_count"] == len(plan["windows"])
    assert plan["windows"], "fixture should yield at least one window"


def test_plan_is_reproducible_across_invocations(
    parquet_file: Path, plan_configs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    main([*_plan_args(parquet_file, plan_configs), "--json"])
    first = capsys.readouterr().out
    main([*_plan_args(parquet_file, plan_configs), "--json"])
    second = capsys.readouterr().out
    assert first == second


def test_plan_requires_attack_config(parquet_file: Path, plan_configs: tuple[Path, Path]) -> None:
    dataset_config, _ = plan_configs
    with pytest.raises(SystemExit) as exit_info:
        main(["plan", "--input", str(parquet_file), "--config", str(dataset_config)])
    assert exit_info.value.code == 2


def test_plan_rejects_unknown_scope(
    parquet_file: Path,
    tmp_path: Path,
    plan_configs: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_config, _ = plan_configs
    bad = tmp_path / "bad.yaml"
    bad.write_text(_SCHEDULING_CONFIG.replace("[validation]", "[nonexistent]"), encoding="utf-8")

    code = main(
        [
            "plan",
            "--input",
            str(parquet_file),
            "--config",
            str(dataset_config),
            "--attack-config",
            str(bad),
        ]
    )
    assert code == EXIT_INPUT_ERROR
    assert "budget_scope" in capsys.readouterr().err


def test_plan_refuses_invalid_dataset(
    tmp_path: Path,
    observations: pd.DataFrame,
    plan_configs: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_config, scheduling_config = plan_configs
    path = tmp_path / "dup.parquet"
    pd.concat([observations, observations.iloc[[0]]]).to_parquet(path, index=False)

    code = main(
        [
            "plan",
            "--input",
            str(path),
            "--config",
            str(dataset_config),
            "--attack-config",
            str(scheduling_config),
        ]
    )
    assert code == EXIT_VALIDATION_FAILED
    assert "fails validation" in capsys.readouterr().err


def test_existing_commands_still_work(parquet_file: Path) -> None:
    # Phase 1 commands remain available and unchanged in behaviour.
    assert main(["validate", "--input", str(parquet_file)]) == EXIT_OK
    assert main(["profile", "--input", str(parquet_file)]) == EXIT_OK


# generate command ------------------------------------------------------------------------

_ATTACK_PARAMS = """
global_seed: 7
quantum: 0.001
spike: {magnitude: 0.2, min_points: 1, max_points: 3}
bias: {magnitude: 0.03}
"""


@pytest.fixture
def attack_params(tmp_path: Path) -> Path:
    path = tmp_path / "attacks.yaml"
    path.write_text(_ATTACK_PARAMS, encoding="utf-8")
    return path


def _generate_args(
    parquet_file: Path, plan_configs: tuple[Path, Path], attack_params: Path
) -> list[str]:
    dataset_config, scheduling_config = plan_configs
    return [
        "generate",
        "--input",
        str(parquet_file),
        "--config",
        str(dataset_config),
        "--attack-config",
        str(scheduling_config),
        "--attack-params",
        str(attack_params),
    ]


def test_generate_text_output(
    parquet_file: Path,
    plan_configs: tuple[Path, Path],
    attack_params: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = _files(tmp_path)
    code = main(_generate_args(parquet_file, plan_configs, attack_params))

    assert code == EXIT_OK
    output = capsys.readouterr().out
    assert "committed" in output
    assert "Clean DataFrame unchanged: True" in output
    assert "No dataset was written" in output
    assert _files(tmp_path) == before, "generate must not write files"


def test_generate_json_output(
    parquet_file: Path,
    plan_configs: tuple[Path, Path],
    attack_params: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([*_generate_args(parquet_file, plan_configs, attack_params), "--json"])

    execution = json.loads(capsys.readouterr().out)["execution"]
    assert code == EXIT_OK
    assert execution["seed"] == 7
    assert execution["frame_unchanged"] is True
    assert execution["committed_total"] <= execution["reserved_before"]
    assert execution["committed_total"] <= execution["requested_quota"]


def test_generate_is_reproducible(
    parquet_file: Path,
    plan_configs: tuple[Path, Path],
    attack_params: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main([*_generate_args(parquet_file, plan_configs, attack_params), "--json"])
    first = capsys.readouterr().out
    main([*_generate_args(parquet_file, plan_configs, attack_params), "--json"])
    second = capsys.readouterr().out
    assert first == second


def test_generate_requires_attack_params(
    parquet_file: Path, plan_configs: tuple[Path, Path]
) -> None:
    dataset_config, scheduling_config = plan_configs
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "generate",
                "--input",
                str(parquet_file),
                "--config",
                str(dataset_config),
                "--attack-config",
                str(scheduling_config),
            ]
        )
    assert exit_info.value.code == 2


def test_generate_rejects_invalid_dataset(
    tmp_path: Path,
    observations: pd.DataFrame,
    plan_configs: tuple[Path, Path],
    attack_params: Path,
) -> None:
    dataset_config, scheduling_config = plan_configs
    path = tmp_path / "dup.parquet"
    pd.concat([observations, observations.iloc[[0]]]).to_parquet(path, index=False)
    code = main(
        [
            "generate",
            "--input",
            str(path),
            "--config",
            str(dataset_config),
            "--attack-config",
            str(scheduling_config),
            "--attack-params",
            str(attack_params),
        ]
    )
    assert code == EXIT_VALIDATION_FAILED


def test_all_commands_available(parquet_file: Path) -> None:
    # Phases 1-3 commands all resolve and keep working.
    assert main(["validate", "--input", str(parquet_file)]) == EXIT_OK
    assert main(["profile", "--input", str(parquet_file)]) == EXIT_OK


@pytest.mark.parametrize("policy", ["skip", "error"])
def test_generate_unsupported_family_policy_is_explicit(
    parquet_file: Path,
    plan_configs: tuple[Path, Path],
    attack_params: Path,
    capsys: pytest.CaptureFixture[str],
    policy: str,
    monkeypatch,
) -> None:
    from attack_engine.attacks.config import EXECUTABLE_FAMILIES

    monkeypatch.setattr(
        "attack_engine.attacks.executor.EXECUTABLE_FAMILIES",
        tuple(f for f in EXECUTABLE_FAMILIES if f is not AttackFamily.STEALTH),
    )
    _, scheduling = plan_configs
    scheduling.write_text(
        _SCHEDULING_CONFIG.replace("[spike, bias]", "[spike, stealth]").replace(
            "bias: {weight: 1.0}", "stealth: {weight: 1.0}"
        ),
        encoding="utf-8",
    )
    code = main(
        [
            *_generate_args(parquet_file, plan_configs, attack_params),
            "--on-unsupported-family",
            policy,
            "--json",
        ]
    )
    captured = capsys.readouterr()
    if policy == "error":
        assert code == EXIT_VALIDATION_FAILED
        assert "non-executable" in captured.err and "stealth" in captured.err
        assert not captured.out
    else:
        assert code == EXIT_OK
        execution = json.loads(captured.out)["execution"]
        assert execution["skipped_families"] == ["stealth"]
        assert execution["skipped_window_count"] > 0
        assert execution["per_family"]["stealth"]["committed"] == 0


def test_generate_refuses_a_source_integrity_mismatch(
    parquet_file: Path,
    plan_configs: tuple[Path, Path],
    attack_params: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            *_generate_args(parquet_file, plan_configs, attack_params),
            "--expected-sha256",
            WRONG_HASH,
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == EXIT_VALIDATION_FAILED
    assert "mismatch" in captured.err.lower()
    assert not captured.out
