"""CLI: public release validation, sensor inventory, schema, and controlled playback."""

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer

from sensorlab.dataset import DetectorDataset, parse_utc
from sensorlab.gate.cli import gate_app
from sensorlab.gate.http_transport import EnvelopeFanout, HTTPSink
from sensorlab.identity import DEFAULT_IDENTITY_DIR, TrustedRegistry
from sensorlab.identity_cli import DEFAULT_PASSWORD_ENV, identity_app, key_password
from sensorlab.playback import PlaybackEngine
from sensorlab.registry import SENSOR_STATIONS, STATION_SENSORS
from sensorlab.signing import (
    SignedEnvelope,
    SignedTelemetrySink,
    SigningService,
    parse_unsigned,
    verify_json,
)
from sensorlab.telemetry import TelemetryMessage
from sensorlab.terminal import TerminalControls
from sensorlab.transport import NDJSONSink

DEFAULT_RELEASE = (
    Path(__file__).resolve().parents[3]
    / "attack-engine/datasets/generated/multivariable-20261023-seed14-acb26f07/combined/detector"
)

app = typer.Typer(
    no_args_is_help=True, help="Six virtual USCRN sensors; public detector data only."
)
sensors = typer.Typer(help="Stable sensor inventory.")
app.add_typer(sensors, name="sensors")
app.add_typer(identity_app, name="identities")
app.add_typer(gate_app, name="gate")
ReleaseOption = Annotated[Path, typer.Option(help="Detector release directory (two public files).")]


@sensors.command("list")
def list_sensors() -> None:
    typer.echo(
        json.dumps(
            [{"sensor_id": key, "station": value} for key, value in SENSOR_STATIONS.items()],
            indent=2,
        )
    )


@app.command()
def schema(
    signed: Annotated[bool, typer.Option(help="Print the signed envelope schema.")] = False,
) -> None:
    """Print telemetry JSON Schema; contains no dataset values."""
    model = SignedEnvelope if signed else TelemetryMessage
    typer.echo(json.dumps(model.model_json_schema(), indent=2))


@app.command()
def validate(release: ReleaseOption = DEFAULT_RELEASE) -> None:
    """Validate all rows, keys, columns, timestamps, and finite-or-missing measurements."""
    try:
        dataset = DetectorDataset.load(release)
    except (OSError, ValueError, TypeError) as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(dataset.summary_json())


@app.command()
def replay(
    release: ReleaseOption = DEFAULT_RELEASE,
    speed: Annotated[float, typer.Option(help="Acceleration factor; 0 means no-wait.")] = 3600,
    station: Annotated[
        list[str] | None, typer.Option(help="Repeat to select station names.")
    ] = None,
    start: Annotated[
        str | None, typer.Option(help="Inclusive observation timestamp, with UTC offset.")
    ] = None,
    end: Annotated[
        str | None, typer.Option(help="Exclusive observation timestamp, with UTC offset.")
    ] = None,
    output: Annotated[Path | None, typer.Option(help="New NDJSON file; never overwrite.")] = None,
    stdout: Annotated[
        bool, typer.Option(help="Also send telemetry to stdout when using --output.")
    ] = False,
    interactive: Annotated[
        bool, typer.Option(help="Enable terminal pause/resume/status/stop.")
    ] = False,
    signed: Annotated[
        bool, typer.Option(help="Wrap each observation in an Ed25519 envelope.")
    ] = False,
    identity_dir: Annotated[
        Path, typer.Option(help="Local encrypted keys and registry.")
    ] = DEFAULT_IDENTITY_DIR,
    password_env: Annotated[
        str, typer.Option(help="Name of the private-key password environment.")
    ] = DEFAULT_PASSWORD_ENV,
    session_id: Annotated[
        UUID | None, typer.Option(help="Explicit preauthorized simulation UUID.")
    ] = None,
    endpoint: Annotated[
        str | None, typer.Option(help="Verified HTTPS Crypto Gate telemetry URL.")
    ] = None,
    ca_cert: Annotated[
        Path | None, typer.Option(help="Trusted development CA; no insecure TLS mode.")
    ] = None,
) -> None:
    """Start a replay. Ctrl+C stops; progress and final counters go to stderr."""
    if interactive and not sys.stdin.isatty():
        raise typer.BadParameter("--interactive requires a terminal on stdin")
    if endpoint and (not signed or session_id is None):
        raise typer.BadParameter("--endpoint requires --signed and an explicit --session-id")
    if ca_cert and not endpoint:
        raise typer.BadParameter("--ca-cert requires --endpoint")
    try:
        dataset = DetectorDataset.load(release).select(
            station, parse_utc(start) if start else None, parse_utc(end) if end else None
        )
        # Validate speed before creating any output artifact.
        engine = PlaybackEngine(dataset, NDJSONSink(sys.stdout), speed=speed, session_id=session_id)
        signing = (
            SigningService.unlock(
                identity_dir,
                key_password(password_env),
                {STATION_SENSORS[station] for station in dataset.stations},
            )
            if signed
            else None
        )
        with ExitStack() as stack:
            http = HTTPSink(endpoint, ca_cert=ca_cert) if endpoint else None
            if http:
                stack.callback(http.close)
            streams = []
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
                streams.append(
                    stack.enter_context(output.open("x", encoding="utf-8", newline="\n"))
                )
            if output is None or stdout:
                streams.append(sys.stdout)
            sink = NDJSONSink(*streams)
            envelopes = EnvelopeFanout(http, sink) if http else sink
            engine.sink = SignedTelemetrySink(signing, envelopes) if signing else sink
            controls = TerminalControls(engine) if interactive else None
            if controls:
                controls.start()
                stack.callback(controls.close)
            result = engine.run()
        typer.echo(json.dumps(result), err=True)
    except (OSError, ValueError, TypeError) as exc:
        typer.echo(f"Replay failed: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("sign")
def sign_file(
    input_file: Path,
    output: Annotated[Path, typer.Option(help="New signed NDJSON output file.")],
    identity_dir: Annotated[
        Path, typer.Option(help="Local encrypted keys and registry.")
    ] = DEFAULT_IDENTITY_DIR,
    password_env: Annotated[
        str, typer.Option(help="Name of the private-key password environment.")
    ] = DEFAULT_PASSWORD_ENV,
) -> None:
    """Sign existing unsigned telemetry, retaining its session, times, sequences and values."""
    count = 0
    try:
        service = SigningService.unlock(identity_dir, key_password(password_env))
        with input_file.open(encoding="utf-8") as source:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as destination:
                sink = SignedTelemetrySink(service, NDJSONSink(destination))
                for line in source:
                    sink.write(parse_unsigned(line))
                    count += 1
        if not count:
            raise ValueError("input contains no telemetry messages")
    except (OSError, ValueError, TypeError) as exc:
        typer.echo(f"Signing failed at line {count + 1}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps({"signed": count, "output": str(output)}))


@app.command("verify")
def verify_file(
    input_file: Path,
    registry: Annotated[
        Path, typer.Option(help="Operator-trusted public registry; no private keys.")
    ] = DEFAULT_IDENTITY_DIR / "registry.json",
) -> None:
    """Verify every envelope locally; does not authorize sessions or reject replays."""
    count = 0
    try:
        trusted = TrustedRegistry.load(registry)
        with input_file.open(encoding="utf-8") as stream:
            for line in stream:
                verify_json(line, trusted)
                count += 1
        if not count:
            raise ValueError("input contains no signed messages")
    except (OSError, ValueError, TypeError) as exc:
        typer.echo(f"Verification failed at line {count + 1}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps({"verified": count, "algorithm": "Ed25519"}))


if __name__ == "__main__":
    app()
