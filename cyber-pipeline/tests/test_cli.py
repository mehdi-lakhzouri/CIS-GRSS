import json
from pathlib import Path

from typer.testing import CliRunner

from sensorlab.cli import app
from sensorlab.telemetry import TelemetryMessage

runner = CliRunner()


def test_inventory_and_schema():
    result = runner.invoke(app, ["sensors", "list"])
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)) == 6
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == TelemetryMessage.model_json_schema()
    schema = Path(__file__).resolve().parents[1] / "schemas/telemetry-v1.schema.json"
    assert json.loads(schema.read_text(encoding="utf-8")) == TelemetryMessage.model_json_schema()


def test_validate_and_file_replay(release, tmp_path):
    result = runner.invoke(app, ["validate", "--release", str(release)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["rows"] == 18
    path = tmp_path / "telemetry.jsonl"
    args = ["replay", "--release", str(release), "--speed", "0", "--output", str(path)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert not result.stdout
    assert json.loads(result.stderr)["transmitted"] == 18
    lines = path.read_text().splitlines()
    assert len(lines) == 18
    assert all(TelemetryMessage.model_validate_json(line) for line in lines)
    snapshot = path.read_bytes()
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert path.read_bytes() == snapshot


def test_stdout_and_filters(release):
    result = runner.invoke(
        app,
        [
            "replay",
            "--release",
            str(release),
            "--speed",
            "0",
            "--station",
            "AZ_Tucson_11_W",
            "--end",
            "2025-01-01T02:00:00Z",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(result.stdout.splitlines()) == 1
    assert json.loads(result.stdout)["station"] == "AZ_Tucson_11_W"
    assert json.loads(result.stderr)["transmitted"] == 1


def test_bad_options_and_interactive_requires_tty(release, tmp_path):
    path = tmp_path / "invalid.jsonl"
    for options in (
        ["--speed", "-1"],
        ["--start", "2025-01-01"],
        ["--station", "unknown"],
        ["--interactive"],
    ):
        result = runner.invoke(
            app, ["replay", "--release", str(release), "--output", str(path), *options]
        )
        assert result.exit_code != 0
        assert not path.exists()


def test_output_cannot_overwrite_source_and_mirror_is_explicit(release, tmp_path):
    source = release / "detector_input.parquet"
    before = source.read_bytes()
    result = runner.invoke(
        app, ["replay", "--release", str(release), "--speed", "0", "--output", str(source)]
    )
    assert result.exit_code == 1
    assert source.read_bytes() == before
    output = tmp_path / "mirrored.jsonl"
    result = runner.invoke(
        app,
        ["replay", "--release", str(release), "--speed", "0", "--output", str(output), "--stdout"],
    )
    assert result.exit_code == 0
    assert result.stdout == output.read_text(encoding="utf-8")
