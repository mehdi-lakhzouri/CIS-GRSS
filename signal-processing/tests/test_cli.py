"""End-to-end CLI: determinism, no-labels operation, refusal to overwrite, provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from signal_processing.cli import app
from signal_processing.features import FEATURE_COLUMNS
from signal_processing.io import sha256_file

runner = CliRunner()


@pytest.fixture
def release(tmp_path: Path, frame: pd.DataFrame) -> Path:
    """A folder holding ONLY detector_input.parquet (no labels anywhere)."""
    folder = tmp_path / "detector"
    folder.mkdir()
    frame.assign(partition="").pipe(_with_partition).to_parquet(folder / "detector_input.parquet")
    return folder / "detector_input.parquet"


def _with_partition(frame: pd.DataFrame) -> pd.DataFrame:
    measured = frame["time"] - pd.Timedelta(hours=1)
    return frame.assign(
        partition=(measured >= "2024-01-01").map({True: "validation", False: "train"})
    )


def _invoke(*args: str) -> str:
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result.output


def test_end_to_end_without_labels(release: Path, tmp_path: Path) -> None:
    assert sorted(p.name for p in release.parent.iterdir()) == ["detector_input.parquet"]
    params = tmp_path / "params" / "frozen_params.json"
    _invoke("fit", "--input", release, "--output", params)
    out = tmp_path / "out"
    _invoke("transform", "--input", release, "--params", params, "--output", out)
    features = pd.read_parquet(out / "signal_features.parquet")
    assert list(features.columns) == ["station", "time", *FEATURE_COLUMNS]
    contract = json.loads((out / "feature_contract.json").read_text(encoding="utf-8"))
    assert list(contract["columns"]) == list(FEATURE_COLUMNS)
    provenance = json.loads((out / "run_provenance.json").read_text(encoding="utf-8"))
    assert provenance["outputs"]["signal_features.parquet"]["sha256"] == sha256_file(
        out / "signal_features.parquet"
    )
    assert provenance["input"]["rows"] == provenance["output_rows"] == len(features)
    for text in (
        (out / "run_provenance.json").read_text(encoding="utf-8"),
        params.read_text(encoding="utf-8"),
    ):
        assert str(tmp_path) not in text and str(Path.home()) not in text


def test_transform_is_byte_deterministic(release: Path, tmp_path: Path) -> None:
    params = tmp_path / "p.json"
    _invoke("fit", "--input", release, "--output", params)
    hashes = []
    for name in ("a", "b"):
        _invoke("transform", "--input", release, "--params", params, "--output", tmp_path / name)
        hashes.append(
            [
                sha256_file(tmp_path / name / f)
                for f in ("signal_features.parquet", "feature_contract.json", "run_provenance.json")
            ]
        )
    assert hashes[0] == hashes[1]


def test_fit_refuses_to_overwrite_different_params(release: Path, tmp_path: Path) -> None:
    params = tmp_path / "p.json"
    _invoke("fit", "--input", release, "--output", params)
    assert "unchanged" in _invoke("fit", "--input", release, "--output", params)
    params.write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["fit", "--input", str(release), "--output", str(params)])
    assert result.exit_code == 1
    assert params.read_text(encoding="utf-8") == "{}"


def test_bench_writes_report(release: Path, tmp_path: Path) -> None:
    params = tmp_path / "p.json"
    _invoke("fit", "--input", release, "--output", params)
    bench = tmp_path / "bench.json"
    _invoke("bench", "--input", release, "--params", params, "--output", bench, "--repeats", "1")
    report = json.loads(bench.read_text(encoding="utf-8"))
    assert report["per_row_us"]["median"] > 0
    assert "node" not in json.dumps(report["machine"])
