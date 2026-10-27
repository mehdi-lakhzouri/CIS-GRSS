"""Command-line interface: ``fit``, ``transform`` and ``bench``.

Exit codes: 0 success, 1 refused (would overwrite different frozen params),
2 invalid input or configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from signal_processing.bench import run_bench
from signal_processing.fit import fit_params
from signal_processing.io import (
    dumps_json,
    read_detector_input,
    sha256_file,
    write_bytes_atomic,
    write_parquet_deterministic,
)
from signal_processing.params import SCHEMA_VERSION, FrozenParams
from signal_processing.partition import load_partition_rules
from signal_processing.provenance import build_feature_contract, build_run_provenance

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

InputOpt = Annotated[
    Path, typer.Option("--input", exists=True, dir_okay=False, help="detector_input.parquet")
]
ParamsOpt = Annotated[
    Path, typer.Option("--params", exists=True, dir_okay=False, help="frozen params JSON")
]


def _fail(message: str, code: int) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


def _load_params(path: Path) -> FrozenParams:
    params = FrozenParams.model_validate_json(path.read_bytes())
    if params.schema_version != SCHEMA_VERSION:
        raise ValueError(f"params schema {params.schema_version} != expected {SCHEMA_VERSION}")
    return params


@app.command()
def fit(
    input_path: InputOpt,
    output: Annotated[Path, typer.Option("--output")] = Path("params/frozen_params.json"),
    partition_config: Annotated[
        Path | None, typer.Option("--partition-config", help="YAML with a 'partitions' list")
    ] = None,
) -> None:
    """Fit frozen parameters on the train partition only."""
    try:
        frame = read_detector_input(input_path)
        rules, source = load_partition_rules(partition_config)
        params = fit_params(frame, sha256_file(input_path), rules, source)
    except (ValueError, KeyError) as exc:
        raise _fail(str(exc), 2) from exc
    payload = dumps_json(params.model_dump(mode="json"))
    if output.exists():
        if output.read_bytes() == payload:
            typer.echo(f"unchanged: {output.name} already holds identical params")
            return
        raise _fail(f"{output} exists with different content; refusing to overwrite", 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(payload, output)
    typer.echo(
        f"wrote {output.name}: {len(params.stations)} stations, {params.train_rows} train rows"
    )
    for station, sp in params.stations.items():
        typer.echo(
            f"  {station}: train replay matches {sp.train_replay_matches}/{sp.train_replay_windows}"
        )


@app.command("transform")
def transform_command(
    input_path: InputOpt,
    params_path: ParamsOpt,
    output: Annotated[Path, typer.Option("--output", file_okay=False, help="output directory")],
) -> None:
    """Compute signal features; writes features, contract and provenance."""
    from signal_processing.features import transform

    try:
        frame = read_detector_input(input_path)
        params = _load_params(params_path)
        features = transform(frame, params)
    except (ValueError, KeyError) as exc:
        raise _fail(str(exc), 2) from exc
    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "signal_features.parquet"
    write_parquet_deterministic(features, features_path)
    params_sha = sha256_file(params_path)
    contract_path = output / "feature_contract.json"
    write_bytes_atomic(dumps_json(build_feature_contract(params, params_sha)), contract_path)
    provenance = build_run_provenance(
        input_name=input_path.name,
        input_sha256=sha256_file(input_path),
        params_name=params_path.name,
        params_sha256=params_sha,
        params=params,
        outputs={p.name: sha256_file(p) for p in (features_path, contract_path)},
        input_rows=len(frame),
        output_rows=len(features),
        stations=sorted(params.stations),
    )
    write_bytes_atomic(dumps_json(provenance), output / "run_provenance.json")
    typer.echo(f"wrote {len(features)} rows x {features.shape[1]} columns to {output.name}/")
    typer.echo(
        f"  signal_features.parquet sha256 {provenance['outputs'][features_path.name]['sha256']}"
    )


@app.command()
def bench(
    input_path: InputOpt,
    params_path: ParamsOpt,
    output: Annotated[Path, typer.Option("--output")] = Path("bench.json"),
    repeats: Annotated[int, typer.Option(min=1)] = 5,
) -> None:
    """Measure transform latency, throughput and peak memory."""
    try:
        frame = read_detector_input(input_path)
        result = run_bench(frame, _load_params(params_path), repeats)
    except (ValueError, KeyError) as exc:
        raise _fail(str(exc), 2) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(dumps_json(result), output)
    typer.echo(
        f"per-row median {result['per_row_us']['median']:.2f} us, "
        f"p95 {result['per_row_us']['p95']:.2f} us, "
        f"{result['rows_per_second']:,.0f} rows/s, "
        f"peak {result['peak_memory_mib_tracemalloc']:.1f} MiB"
    )


if __name__ == "__main__":
    app()
