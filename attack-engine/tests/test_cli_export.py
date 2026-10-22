from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from attack_engine.cli import EXIT_OK, EXIT_VALIDATION_FAILED, main
from attack_engine.utils.hashing import sha256_file

_DATASET_CONFIG = """
validation_profile:
  name: fixture-export
partitions:
  - {name: train, start: '2022-01-01T00:00:00Z', end: '2024-01-01T00:00:00Z'}
  - {name: validation, start: '2024-01-01T00:00:00Z', end: '2025-01-01T00:00:00Z'}
  - {name: test, start: '2025-01-01T00:00:00Z', end: '2026-01-01T00:00:00Z'}
"""

_SCHEDULING_CONFIG = """
global_seed: 7
attack_budget: 0.3
budget_scope: [validation]
enabled_families: [spike, bias]
min_window_length: 2
max_window_length: 6
max_attempts: 500
family_parameters:
  spike: {weight: 1.0}
  bias: {weight: 1.0}
"""

_ATTACK_PARAMS = """
global_seed: 7
quantum: 0.001
spike: {magnitude: 0.2, min_points: 1, max_points: 3}
bias: {magnitude: 0.03}
"""


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return workdir


@pytest.fixture
def export_configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(_DATASET_CONFIG, encoding="utf-8")
    scheduling_config = tmp_path / "scheduling.yaml"
    scheduling_config.write_text(_SCHEDULING_CONFIG, encoding="utf-8")
    attack_params = tmp_path / "attacks.yaml"
    attack_params.write_text(_ATTACK_PARAMS, encoding="utf-8")
    return dataset_config, scheduling_config, attack_params


def _export_args(
    parquet_file: Path, export_configs: tuple[Path, Path, Path], output: Path, **extra: str
) -> list[str]:
    dataset_config, scheduling_config, attack_params = export_configs
    args = [
        "export",
        "--input",
        str(parquet_file),
        "--config",
        str(dataset_config),
        "--attack-config",
        str(scheduling_config),
        "--attack-params",
        str(attack_params),
        "--output",
        str(output),
    ]
    for key, value in extra.items():
        args += [f"--{key.replace('_', '-')}", value]
    return args


def test_export_creates_a_run_directory(
    parquet_file: Path, export_configs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "generated"
    digest_before = sha256_file(parquet_file)

    code = main(_export_args(parquet_file, export_configs, output, run_id="run-cli"))

    assert code == EXIT_OK
    run_dir = output / "run-cli"
    assert run_dir.is_dir()
    for name in (
        "uscrn_attacked.parquet",
        "detector_input.parquet",
        "ground_truth.parquet",
        "attack_manifest.csv",
        "run_provenance.json",
        "data_contract.json",
        "SHA256SUMS.txt",
        "HANDOVER.md",
    ):
        assert (run_dir / name).is_file()
    assert sha256_file(parquet_file) == digest_before, "the CLI must not modify the source"


def test_export_json_output(
    parquet_file: Path, export_configs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "generated"
    code = main([*_export_args(parquet_file, export_configs, output, run_id="run-json"), "--json"])
    assert code == EXIT_OK


def test_export_refuses_to_overwrite_existing_run(
    parquet_file: Path, export_configs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "generated"
    args = _export_args(parquet_file, export_configs, output, run_id="run-dup")

    assert main(args) == EXIT_OK
    second_code = main(args)

    assert second_code != EXIT_OK


def test_export_requires_attack_params(
    parquet_file: Path, export_configs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    dataset_config, scheduling_config, _ = export_configs
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "export",
                "--input",
                str(parquet_file),
                "--config",
                str(dataset_config),
                "--attack-config",
                str(scheduling_config),
                "--output",
                str(tmp_path / "generated"),
            ]
        )
    assert exit_info.value.code == 2


def test_export_rejects_invalid_dataset(
    tmp_path: Path,
    observations: pd.DataFrame,
    export_configs: tuple[Path, Path, Path],
) -> None:
    path = tmp_path / "dup.parquet"
    pd.concat([observations, observations.iloc[[0]]]).to_parquet(path, index=False)

    code = main(_export_args(path, export_configs, tmp_path / "generated"))

    assert code == EXIT_VALIDATION_FAILED


def test_export_reproducible_ground_truth_across_runs(
    parquet_file: Path, export_configs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "generated"
    main(_export_args(parquet_file, export_configs, output, run_id="run-a"))
    main(_export_args(parquet_file, export_configs, output, run_id="run-b"))

    first = pd.read_parquet(output / "run-a" / "ground_truth.parquet")
    second = pd.read_parquet(output / "run-b" / "ground_truth.parquet")
    pd.testing.assert_frame_equal(first, second)


def test_export_missing_output_does_not_raise_file_not_found(
    parquet_file: Path, export_configs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    deep_output = tmp_path / "a" / "b" / "c"
    code = main(_export_args(parquet_file, export_configs, deep_output, run_id="run-deep"))
    assert code == EXIT_OK
    assert (deep_output / "run-deep").is_dir()
