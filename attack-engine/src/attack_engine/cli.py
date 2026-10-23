"""Command-line interface. Parses arguments, wires project modules, and formats output.

Exit codes: 0 success, 1 validation failure or integrity mismatch, 2 usage, input or
configuration error. ``calibrate`` writes a JSON reference artifact; ``export`` writes a
full dataset bundle (Phase 5). Validation, profiling, planning and generation never write
datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from attack_engine import __version__
from attack_engine.attacks.calibration import CalibrationError, fit_calibration, save_calibration
from attack_engine.attacks.config import AttackConfigError, load_attack_config
from attack_engine.attacks.executor import (
    AttackExecutor,
    ExecutionError,
    ExecutionSummary,
    UnsupportedFamilyPolicy,
)
from attack_engine.config import ConfigError, load_config
from attack_engine.dataset.loader import DatasetError, DatasetLoader, LoadedDataset
from attack_engine.dataset.profiler import DatasetProfile, DatasetProfiler, ProfilingError
from attack_engine.dataset.schema import (
    FIXTURE_PROFILE,
    DatasetSchema,
    PartitionPlan,
    ValidationProfile,
    uscrn_hourly_schema,
)
from attack_engine.dataset.validator import DatasetValidator, ValidationReport
from attack_engine.exporting.models import ExportConfiguration, ExportError, ExportResult
from attack_engine.exporting.pipeline import export_run
from attack_engine.scheduling.config import ConfigError as SchedulingConfigError
from attack_engine.scheduling.config import SchedulingConfig, load_scheduling_config
from attack_engine.scheduling.scheduler import (
    AttackWindowScheduler,
    ScheduleResult,
    SchedulingError,
)
from attack_engine.utils.hashing import (
    IntegrityResult,
    IntegrityStatus,
    compare_sha256,
    normalize_sha256,
)

EXIT_OK = 0
EXIT_VALIDATION_FAILED = 1
EXIT_INPUT_ERROR = 2

DEFAULT_CONFIG_PATH = Path("configs") / "dataset.yaml"
PROFILE_CHOICES = ("real", "fixture")
MAX_LISTED_WINDOWS = 8


@dataclass(frozen=True)
class _Settings:
    profile: ValidationProfile
    partitions: PartitionPlan | None
    expected_sha256: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attack-engine",
        description=(
            "Attack Engine (TSYP14, Track 2: False Data Injection). Provides dataset validation, "
            "profiling, attack-window planning and in-memory attack generation."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, summary in (
        ("validate", "Validate a dataset against the schema and validation profile."),
        ("profile", "Print descriptive statistics of a dataset."),
    ):
        command = commands.add_parser(name, help=summary, description=summary)
        _add_common_arguments(command)

    plan_summary = (
        "Plan attack windows and budget allocation (Phase 2). Reserves where attacks "
        "could go; does not modify or write any observation value."
    )
    plan = commands.add_parser("plan", help=plan_summary, description=plan_summary)
    _add_common_arguments(plan)
    plan.add_argument(
        "--attack-config",
        required=True,
        type=Path,
        metavar="PATH",
        help="YAML scheduling configuration (seed, budget, families).",
    )

    generate_summary = (
        "Execute attacks over a plan in memory and print a summary (Phases 3-4). "
        "Computes candidate values and commits actual modified counts; writes no files."
    )
    generate = commands.add_parser("generate", help=generate_summary, description=generate_summary)
    _add_common_arguments(generate)
    generate.add_argument(
        "--attack-config",
        required=True,
        type=Path,
        metavar="PATH",
        help="YAML scheduling configuration (seed, budget, families).",
    )
    generate.add_argument(
        "--attack-params",
        required=True,
        type=Path,
        metavar="PATH",
        help="YAML attack parameters (magnitudes, signs, quantization).",
    )
    generate.add_argument(
        "--on-unsupported-family",
        choices=[policy.value for policy in UnsupportedFamilyPolicy],
        default=UnsupportedFamilyPolicy.SKIP.value,
        help=(
            "Policy when the plan contains a future unsupported family: "
            "'skip' releases its reservations and reports them (default); 'error' fails the run."
        ),
    )
    calibrate = commands.add_parser(
        "calibrate", help="Fit train-only station context references and write one JSON artifact."
    )
    _add_common_arguments(calibrate)
    calibrate.add_argument("--attack-params", required=True, type=Path, metavar="PATH")
    calibrate.add_argument("--output", required=True, type=Path, metavar="PATH")

    export_summary = (
        "Export ground truth, provenance and the detector/audit datasets (Phase 5), "
        "downstream of an already-executed attack plan. Writes a new, uniquely "
        "identified run directory; never overwrites an existing one."
    )
    export = commands.add_parser("export", help=export_summary, description=export_summary)
    _add_common_arguments(export)
    export.add_argument("--attack-config", required=True, type=Path, metavar="PATH")
    export.add_argument("--attack-params", required=True, type=Path, metavar="PATH")
    export.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Directory that will contain one new <run-id> subdirectory.",
    )
    export.add_argument(
        "--run-id", metavar="ID", help="Explicit run identifier (default: generated, timestamped)."
    )
    export.add_argument(
        "--on-unsupported-family",
        choices=[policy.value for policy in UnsupportedFamilyPolicy],
        default=UnsupportedFamilyPolicy.SKIP.value,
    )
    return parser


def _add_common_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--input", required=True, type=Path, metavar="PATH", help="Parquet file (read only)."
    )
    command.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help=f"YAML configuration (default: {DEFAULT_CONFIG_PATH} if it exists).",
    )
    command.add_argument(
        "--validation-profile",
        choices=PROFILE_CHOICES,
        help=(
            "'real' applies the reference-delivery expectations, expected hash and partitions "
            "from the configuration; 'fixture' applies generic schema rules only. "
            "Default: 'real' when a configuration is available, otherwise 'fixture'."
        ),
    )
    command.add_argument(
        "--expected-sha256", metavar="HEX", help="Expected SHA-256 of the input (overrides config)."
    )
    command.add_argument("--json", action="store_true", help="Print JSON instead of text.")


def _resolve_settings(args: argparse.Namespace) -> _Settings:
    config_path = args.config
    if config_path is None and DEFAULT_CONFIG_PATH.is_file():
        config_path = DEFAULT_CONFIG_PATH
    config = load_config(config_path) if config_path is not None else None

    choice = args.validation_profile or ("real" if config is not None else "fixture")
    if choice == "real":
        if config is None:
            raise ConfigError("the 'real' validation profile requires a configuration (--config)")
        settings = _Settings(config.validation_profile, config.partitions, config.expected_sha256)
    else:
        settings = _Settings(FIXTURE_PROFILE, None, None)

    if args.expected_sha256 is not None:
        try:
            expected = normalize_sha256(args.expected_sha256)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        settings = _Settings(settings.profile, settings.partitions, expected)
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema = uscrn_hourly_schema()
    try:
        settings = _resolve_settings(args)
        dataset = DatasetLoader(schema=schema).load(args.input)
    except (ConfigError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    integrity = compare_sha256(dataset.source_sha256, settings.expected_sha256)
    report = DatasetValidator(schema, settings.profile, settings.partitions).validate(dataset.frame)

    if args.command == "validate":
        if args.json:
            _print_json(dataset, integrity, validation=report.model_dump(mode="json"))
        else:
            print(_format_validation(dataset, integrity, report))
        return EXIT_OK if report.passed and integrity.ok else EXIT_VALIDATION_FAILED

    if args.command == "plan":
        return _run_plan(args, schema, settings, dataset, integrity, report)

    if args.command == "generate":
        return _run_generate(args, schema, settings, dataset, integrity, report)

    if args.command == "calibrate":
        return _run_calibrate(args, schema, settings, dataset, integrity, report)

    if args.command == "export":
        return _run_export(args, schema, settings, dataset, integrity, report)

    try:
        profile = DatasetProfiler(schema).profile(dataset.frame, report)
    except ProfilingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_FAILED
    if args.json:
        _print_json(dataset, integrity, profile=profile.model_dump(mode="json"))
    else:
        print(_format_profile(dataset, integrity, profile))
    return EXIT_OK


def _run_plan(
    args: argparse.Namespace,
    schema: DatasetSchema,
    settings: _Settings,
    dataset: LoadedDataset,
    integrity: IntegrityResult,
    report: ValidationReport,
) -> int:
    if not report.passed:
        print(
            "error: cannot plan on a dataset that fails validation; run 'validate' first",
            file=sys.stderr,
        )
        return EXIT_VALIDATION_FAILED
    if settings.partitions is None:
        print(
            "error: planning requires partition definitions; use the real validation profile "
            "with a dataset --config that declares partitions",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR
    try:
        attack_config = load_scheduling_config(args.attack_config)
        _check_scope(attack_config, settings.partitions)
        result = AttackWindowScheduler(schema, settings.partitions, attack_config).plan(
            dataset.frame
        )
    except SchedulingConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except SchedulingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_FAILED

    if args.json:
        _print_json(dataset, integrity, plan=_plan_document(result))
    else:
        print(_format_plan(dataset, integrity, result))
    return EXIT_OK


def _check_scope(config: SchedulingConfig, partitions: PartitionPlan) -> None:
    known = {partition.name for partition in partitions.partitions}
    unknown = [name for name in config.budget_scope if name not in known]
    if unknown:
        raise SchedulingConfigError(
            f"budget_scope names not in the partition plan {sorted(known)}: {unknown}"
        )


def _run_generate(
    args: argparse.Namespace,
    schema: DatasetSchema,
    settings: _Settings,
    dataset: LoadedDataset,
    integrity: IntegrityResult,
    report: ValidationReport,
) -> int:
    if not report.passed or not integrity.ok:
        print(
            "error: cannot generate attacks on an invalid dataset or a SHA-256 mismatch; "
            "run 'validate' first",
            file=sys.stderr,
        )
        return EXIT_VALIDATION_FAILED
    if settings.partitions is None:
        print(
            "error: attack generation requires partition definitions; use the real validation "
            "profile with a dataset --config that declares partitions",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR
    try:
        scheduling_config = load_scheduling_config(args.attack_config)
        _check_scope(scheduling_config, settings.partitions)
        attack_config = load_attack_config(args.attack_params)
        plan = AttackWindowScheduler(schema, settings.partitions, scheduling_config).plan(
            dataset.frame
        )
        executor = AttackExecutor(
            schema,
            settings.partitions,
            attack_config,
            unsupported_policy=UnsupportedFamilyPolicy(args.on_unsupported_family),
        )
        summary = executor.execute(dataset.frame, plan)
    except (SchedulingConfigError, AttackConfigError, CalibrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except (SchedulingError, ExecutionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_FAILED

    if args.json:
        _print_json(dataset, integrity, execution=_execution_document(plan, summary))
    else:
        print(_format_execution(dataset, integrity, plan, summary))
    return EXIT_OK


def _run_calibrate(
    args: argparse.Namespace,
    schema: DatasetSchema,
    settings: _Settings,
    dataset: LoadedDataset,
    integrity: IntegrityResult,
    report: ValidationReport,
) -> int:
    if not report.passed or not integrity.ok:
        print("error: calibration requires a valid dataset and matching integrity", file=sys.stderr)
        return EXIT_VALIDATION_FAILED
    try:
        config = load_attack_config(args.attack_params)
        if config.stealth is None or settings.partitions is None:
            raise CalibrationError("calibration requires stealth parameters and train partitions")
        artifact = fit_calibration(dataset.frame, schema, settings.partitions, config.stealth)
        save_calibration(artifact, args.output)
    except (AttackConfigError, CalibrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    document = artifact.model_dump(mode="json")
    if args.json:
        _print_json(dataset, integrity, calibration=document)
    else:
        print(
            f"Calibration: {args.output}\nSHA-256: {artifact.calibration_sha256}\n"
            f"Train only: {artifact.partition_start.isoformat()} -> "
            f"{artifact.partition_end.isoformat()}\n"
            f"Station models: {', '.join(artifact.station_models)}"
        )
    return EXIT_OK


def _run_export(
    args: argparse.Namespace,
    schema: DatasetSchema,
    settings: _Settings,
    dataset: LoadedDataset,
    integrity: IntegrityResult,
    report: ValidationReport,
) -> int:
    if not report.passed or not integrity.ok:
        print(
            "error: cannot export from an invalid dataset or a SHA-256 mismatch; "
            "run 'validate' first",
            file=sys.stderr,
        )
        return EXIT_VALIDATION_FAILED
    if settings.partitions is None:
        print(
            "error: export requires partition definitions; use the real validation profile "
            "with a dataset --config that declares partitions",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR
    try:
        scheduling_config = load_scheduling_config(args.attack_config)
        _check_scope(scheduling_config, settings.partitions)
        attack_config = load_attack_config(args.attack_params)
        plan = AttackWindowScheduler(schema, settings.partitions, scheduling_config).plan(
            dataset.frame
        )
        executor = AttackExecutor(
            schema,
            settings.partitions,
            attack_config,
            unsupported_policy=UnsupportedFamilyPolicy(args.on_unsupported_family),
        )
        summary = executor.execute(dataset.frame, plan)
        export_config = ExportConfiguration(output_root=args.output, run_id=args.run_id)
        result = export_run(
            dataset=dataset,
            schema=schema,
            partitions=settings.partitions,
            scheduling_config_path=args.attack_config,
            attack_config=attack_config,
            attack_params_path=args.attack_params,
            plan=plan,
            summary=summary,
            config=export_config,
        )
    except (SchedulingConfigError, AttackConfigError, CalibrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except (SchedulingError, ExecutionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_FAILED
    except (ExportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_FAILED

    if args.json:
        _print_json(dataset, integrity, export=_export_document(result))
    else:
        print(_format_export(dataset, integrity, result))
    return EXIT_OK


def _export_document(result: ExportResult) -> dict[str, Any]:
    document = result.model_dump(mode="json")
    document["run_dir"] = str(result.run_dir)
    return document


def _format_export(dataset: LoadedDataset, integrity: IntegrityResult, result: ExportResult) -> str:
    lines = [
        *_header(dataset, integrity),
        f"Run     : {result.run_id}",
        f"Output  : {result.run_dir}",
        f"Rows    : {result.row_count} ({result.finite_target_count} finite target, "
        f"{result.missing_target_count} missing target)",
        f"Modified: {result.modified_count} observations ({result.manifest_row_count} "
        "manifest events)",
        f"Cells   : {result.modified_cells}; per variable: {result.per_variable_modifications}",
        f"Budget  : {result.budget_unit}; quota deficit {result.quota_deficit}",
        f"Release : {result.detector_release_dir} (share only this directory)",
    ]
    for family, committed in sorted(result.per_family_committed.items()):
        lines.append(f"  {family:8s} committed {committed:>6d}")
    lines.append(f"Validation: {'PASSED' if result.validation_passed else 'FAILED'}")
    lines += [f"  {message}" for message in result.validation_messages]
    lines.append("Files (SHA-256):")
    lines += [f"  {name}: {digest}" for name, digest in sorted(result.file_sha256.items())]
    return "\n".join(lines)


# Presentation --------------------------------------------------------------------------


def _print_json(dataset: LoadedDataset, integrity: IntegrityResult, **sections: Any) -> None:
    document = {
        "dataset": dataset.metadata.model_dump(mode="json"),
        "integrity": integrity.model_dump(mode="json"),
        **sections,
    }
    print(json.dumps(document, indent=2))


def _format_integrity(integrity: IntegrityResult) -> str:
    if integrity.status is IntegrityStatus.MATCH:
        return f"{integrity.actual_sha256} (matches expected)"
    if integrity.status is IntegrityStatus.MISMATCH:
        return f"{integrity.actual_sha256} (MISMATCH: expected {integrity.expected_sha256})"
    return f"{integrity.actual_sha256} (no expected hash configured)"


def _header(dataset: LoadedDataset, integrity: IntegrityResult) -> list[str]:
    metadata = dataset.metadata
    return [
        f"Dataset : {metadata.path}",
        f"Size    : {metadata.size_bytes} bytes, {metadata.row_count} rows, "
        f"{metadata.column_count} columns",
        f"SHA-256 : {_format_integrity(integrity)}",
    ]


def _format_validation(
    dataset: LoadedDataset, integrity: IntegrityResult, report: ValidationReport
) -> str:
    status = "PASSED" if report.passed else "FAILED"
    lines = [
        *_header(dataset, integrity),
        f"Profile : {report.profile_name}",
        f"Result  : {status} ({len(report.errors)} errors, {len(report.warnings)} warnings)",
    ]
    lines += [
        f"  [{issue.severity.value.upper()}] {issue.code}: {issue.message}"
        for issue in report.issues
    ]
    if report.partition_row_counts:
        counts = ", ".join(f"{name}={n}" for name, n in report.partition_row_counts.items())
        lines.append(f"Partitions: {counts}")
    if not integrity.ok:
        lines.append("Integrity check FAILED: the input is not the expected file.")
    return "\n".join(lines)


def _format_profile(
    dataset: LoadedDataset, integrity: IntegrityResult, profile: DatasetProfile
) -> str:
    target = profile.target
    lines = [*_header(dataset, integrity), f"Stations: {profile.station_count}"]
    if profile.observation_period is not None:
        period = profile.observation_period
        lines.append(
            f"Period  : {period.first_observation_end.isoformat()} to "
            f"{period.last_observation_end.isoformat()} (interval-end UTC)"
        )
    lines.append(
        f"Target  : {target.column} [{target.unit}] coverage {target.coverage_percent:.4f}% "
        f"({target.missing_count} missing)"
    )
    if target.finite_count:
        lines.append(
            f"          min {target.minimum} max {target.maximum} "
            f"mean {target.mean:.6f} std {target.std}"
        )
    lines.append(
        f"Keys    : {profile.duplicate_key_count} duplicates; gaps: "
        f"{profile.temporal_gaps.gap_count} ({profile.temporal_gaps.missing_intervals} missing "
        f"intervals); year-boundary rows: {profile.year_boundary_row_count}"
    )
    lines.append("Per station (rows, target missing, coverage %):")
    lines += [
        f"  {name}: {station.row_count}, {station.target_missing_count}, "
        f"{station.target_coverage_percent:.4f}"
        for name, station in profile.stations.items()
    ]
    if profile.partition_row_counts:
        counts = ", ".join(f"{name}={n}" for name, n in profile.partition_row_counts.items())
        lines.append(f"Partitions: {counts}")
    if profile.validation is not None:
        validation = profile.validation
        lines.append(
            f"Validation: {'PASSED' if validation.passed else 'FAILED'} "
            f"({validation.error_count} errors, {validation.warning_count} warnings)"
        )
        lines += [f"  [ERROR] {message}" for message in validation.errors]
        lines += [f"  [WARNING] {message}" for message in validation.warnings]
    return "\n".join(lines)


def _plan_document(result: ScheduleResult) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "scope": list(result.scope),
        "families": [family.value for family in result.families],
        "requested_budget": result.requested_budget,
        "finite_denominator": result.finite_denominator,
        "requested_quota": result.requested_quota,
        "allocated_total": result.ledger.total_allocated,
        "reserved_total": result.reserved_total,
        "committed_total": result.ledger.total_committed,
        "unfulfilled_total": result.unfulfilled_total,
        "family_quotas": {family.value: quota for family, quota in result.family_quotas.items()},
        "denominators": {cell.label(): count for cell, count in result.denominators.items()},
        "cell_ledgers": [
            {
                "family": ledger.family.value,
                "cell": ledger.cell.label(),
                "finite_denominator": ledger.finite_denominator,
                "allocated": ledger.allocated,
                "reserved": ledger.reserved,
                "committed": ledger.committed,
                "unfulfilled": ledger.unfulfilled,
            }
            for ledger in result.ledger.cell_ledgers()
            if ledger.allocated > 0
        ],
        "window_count": len(result.windows),
        "windows": [window.model_dump(mode="json") for window in result.windows],
        "rejections": [rejection.model_dump(mode="json") for rejection in result.rejections],
    }


def _format_plan(dataset: LoadedDataset, integrity: IntegrityResult, result: ScheduleResult) -> str:
    lines = [
        *_header(dataset, integrity),
        f"Seed    : {result.seed}",
        f"Scope   : {', '.join(result.scope)}",
        f"Stations: {sorted({cell.station for cell in result.denominators})}",
        f"Budget  : {result.requested_budget} of {result.finite_denominator} finite targets "
        f"= {result.requested_quota} observations requested",
        f"Allocated {result.ledger.total_allocated}, reserved {result.reserved_total}, "
        f"unfulfilled {result.unfulfilled_total}, actually modified unknown until execution",
    ]
    lines.append("Family quotas (allocated -> reserved):")
    for family in result.families:
        allocated = sum(
            ledger.allocated for ledger in result.ledger.cell_ledgers() if ledger.family is family
        )
        reserved = sum(
            ledger.reserved for ledger in result.ledger.cell_ledgers() if ledger.family is family
        )
        windows = len(result.windows_for(family))
        lines.append(
            f"  {family.value:8s} quota {result.family_quotas[family]:>6d}  "
            f"allocated {allocated:>6d}  reserved {reserved:>6d}  windows {windows:>4d}"
        )
    lines.append(f"Planned windows: {len(result.windows)} (showing up to {MAX_LISTED_WINDOWS})")
    for window in result.windows[:MAX_LISTED_WINDOWS]:
        lines.append(
            f"  {window.window_id}: {window.station}/{window.partition} len {window.length} "
            f"[{window.first_observation_time.isoformat()} .. "
            f"{window.end_exclusive.isoformat()})"
        )
    if result.rejections:
        lines.append(f"Unfulfilled / rejected ({len(result.rejections)}):")
        for rejection in result.rejections[:MAX_LISTED_WINDOWS]:
            family = rejection.family.value if rejection.family else "-"
            lines.append(
                f"  [{rejection.reason.value}] {family} {rejection.station}/{rejection.partition}"
                f" (x{rejection.count}): {rejection.detail}"
            )
    lines.append("No observation values were modified; no dataset was written.")
    return "\n".join(lines)


def _execution_document(plan: ScheduleResult, summary: ExecutionSummary) -> dict[str, Any]:
    from collections import Counter

    rejection_reasons = Counter(
        result.rejection_reason.value
        for result in summary.rejected_results
        if result.rejection_reason is not None
    )
    return {
        "seed": summary.seed,
        "requested_quota": summary.requested_quota,
        "reserved_before": summary.reserved_before,
        "committed_total": summary.committed_total,
        "committed_cells": summary.committed_cells,
        "committed_rows": summary.committed_rows,
        "budget_unit": summary.budget_unit,
        "max_modified_rows": plan.max_modified_rows,
        "max_modified_cells": plan.max_modified_cells,
        "per_variable_modifications": summary.per_variable_modifications,
        "released_total": summary.released_total,
        "deficit_vs_quota": summary.deficit_vs_quota,
        "max_modifiable_total": summary.max_modifiable_total,
        "frame_unchanged": summary.frame_unchanged,
        "calibration_sha256": summary.calibration_sha256,
        "context_coverage": {
            "available_rows": summary.context_available,
            "total_rows": summary.context_total,
            "fraction": summary.context_available / summary.context_total
            if summary.context_total
            else None,
        },
        "skipped_families": [family.value for family in summary.skipped_families],
        "skipped_window_count": summary.skipped_window_count,
        "per_family": {
            family.value: {
                "planned_windows": execution.planned_windows,
                "applied_windows": execution.applied_windows,
                "rejected_windows": execution.rejected_windows,
                "skipped_windows": execution.skipped_windows,
                "allocated": execution.allocated,
                "unreserved": execution.unreserved,
                "reserved": execution.reserved,
                "max_modifiable": execution.max_modifiable,
                "committed": execution.committed,
                "modified_rows": execution.modified_rows,
                "modified_cells": execution.modified_cells,
                "per_variable_modifications": execution.per_variable_modifications,
                "deficit": execution.deficit,
                "structural_deficit": execution.structural_deficit,
                "unfulfilled_quota": execution.unfulfilled_quota,
                "rejection_reasons": dict(
                    Counter(
                        result.rejection_reason.value
                        for result in summary.rejected_results
                        if result.family is family and result.rejection_reason is not None
                    )
                ),
            }
            for family, execution in summary.per_family.items()
        },
        "rejection_reasons": dict(rejection_reasons),
        "applied_event_count": len(summary.applied_results),
    }


def _format_execution(
    dataset: LoadedDataset,
    integrity: IntegrityResult,
    plan: ScheduleResult,
    summary: ExecutionSummary,
) -> str:
    lines = [
        *_header(dataset, integrity),
        f"Seed    : {summary.seed}",
        f"Budget  : requested {summary.requested_quota}, reserved {summary.reserved_before}, "
        f"committed {summary.committed_total}, released {summary.released_total}, "
        f"deficit vs quota {summary.deficit_vs_quota}",
        f"Modified: {summary.committed_cells} cells across {summary.committed_rows} rows "
        f"(budget unit: {summary.budget_unit})",
        f"Per variable: {summary.per_variable_modifications}",
    ]
    if summary.skipped_families:
        skipped = ", ".join(family.value for family in summary.skipped_families)
        lines.append(
            f"Skipped (planned but not executable this phase): {skipped} "
            f"({summary.skipped_window_count} windows, reservations released)"
        )
    if summary.context_total:
        lines.append(
            f"Context coverage: {summary.context_available}/{summary.context_total} "
            "planned Stealth observations (finite required measurements)"
        )
        lines.append(f"Calibration SHA-256: {summary.calibration_sha256}")
    lines.append("Per family (windows applied/planned, reserved -> max achievable -> committed):")
    for family, execution in summary.per_family.items():
        lines.append(
            f"  {family.value:8s} windows {execution.applied_windows}/{execution.planned_windows}"
            f"  reserved {execution.reserved:>6d}  max {execution.max_modifiable:>6d}"
            f"  committed {execution.committed:>6d}  deficit {execution.deficit:>5d}"
            f"  unreserved {execution.unreserved:>4d}"
            f"  unfulfilled quota {execution.unfulfilled_quota:>5d}"
        )
    applied = summary.applied_results
    lines.append(f"Applied events: {len(applied)}, rejected: {len(summary.rejected_results)}")
    from collections import Counter

    reasons = Counter(
        result.rejection_reason.value
        for result in summary.rejected_results
        if result.rejection_reason is not None
    )
    for reason, count in reasons.most_common():
        lines.append(f"  rejected [{reason}]: {count}")
    lines.append(
        f"Clean DataFrame unchanged: {summary.frame_unchanged}. "
        "Reserved != committed: reservations are capacity, committed are actual modifications."
    )
    lines.append("No dataset was written; export and manifests belong to a later phase.")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
