"""Export and validate six independent USCRN runs without overwriting earlier bundles.

Run from attack-engine: uv run python scripts/export_multivariable.py
Only each run's detector/ subdirectory is suitable for release to the AI team.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from validate_multivariable import execution_digest

from attack_engine.attacks.config import load_attack_config
from attack_engine.attacks.executor import AttackExecutor
from attack_engine.config import load_config
from attack_engine.dataset.loader import DatasetLoader
from attack_engine.dataset.schema import uscrn_hourly_schema
from attack_engine.exporting.models import ExportConfiguration, ExportError
from attack_engine.exporting.pipeline import export_run
from attack_engine.exporting.writer import (
    hash_files,
    run_in_temp_directory,
    write_json,
    write_sha256sums,
)
from attack_engine.scheduling.config import load_scheduling_config
from attack_engine.scheduling.scheduler import AttackWindowScheduler
from attack_engine.utils.hashing import sha256_file

PROJECT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ("spike", "bias", "drift", "replay", "stealth", "combined")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT.parent / "USCRN_livraison/data/uscrn_clean.parquet"
    )
    parser.add_argument("--output", type=Path, default=PROJECT / "datasets/generated")
    parser.add_argument("--group-id")
    parser.add_argument("--campaigns", nargs="+", choices=CAMPAIGNS, default=list(CAMPAIGNS))
    args = parser.parse_args()
    if len(set(args.campaigns)) != len(args.campaigns):
        parser.error("campaigns must be unique")
    group_id = (
        args.group_id
        or f"multivariable-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-seed14-{uuid4().hex[:8]}"
    )
    ExportConfiguration(output_root=args.output, run_id=group_id)
    final_group = (args.output / group_id).resolve()
    protected_path = PROJECT / "validation/multivariable_export_integrity_before.json"
    protected = json.loads(protected_path.read_text(encoding="utf-8"))

    def verify_originals():
        changed = [
            name
            for name, digest in protected.items()
            if sha256_file(PROJECT.parent / name) != digest
        ]
        if changed:
            raise ExportError(
                f"protected source/prior-export/configuration files changed: {changed}"
            )

    verify_originals()
    project_config = load_config(PROJECT / "configs/dataset.yaml")
    schema = uscrn_hourly_schema()
    dataset = DatasetLoader(schema).load(args.input)
    if dataset.source_sha256 != project_config.expected_sha256:
        raise ExportError("this reference execution requires the validated USCRN delivery")
    reference = json.loads(
        (PROJECT / "validation/multivariable_all_families_results.json").read_text(encoding="utf-8")
    )
    snapshot = dataset.frame.copy(deep=True)
    evidence = {}

    def write_group(staging):
        for campaign in args.campaigns:
            root = PROJECT / "configs/campaigns" / campaign
            schedule_path, params_path = root / "scheduling.yaml", root / "attacks.yaml"
            scheduling, parameters = (
                load_scheduling_config(schedule_path),
                load_attack_config(params_path),
            )
            scheduler = AttackWindowScheduler(schema, project_config.partitions, scheduling)
            executor = AttackExecutor(schema, project_config.partitions, parameters)
            plan = scheduler.plan(dataset.frame)
            summary = executor.execute(dataset.frame, plan)
            repeated = executor.execute(dataset.frame, scheduler.plan(dataset.frame))
            if execution_digest(summary) != execution_digest(repeated):
                raise ExportError(f"execution is not reproducible for {campaign}")
            if not all(item.applied_windows for item in summary.per_family.values()):
                raise ExportError(f"at least one family has no applied events for {campaign}")
            if campaign in reference["campaigns"]:
                known = reference["campaigns"][campaign]
                if (known["modified_cells"], known["modified_rows"]) != (
                    summary.committed_cells,
                    summary.committed_rows,
                ):
                    raise ExportError(f"validated generation counts changed for {campaign}")
            result = export_run(
                dataset=dataset,
                schema=schema,
                partitions=project_config.partitions,
                scheduling_config_path=schedule_path,
                attack_config=parameters,
                attack_params_path=params_path,
                plan=plan,
                summary=summary,
                config=ExportConfiguration(output_root=staging, run_id=campaign),
            )
            provenance = json.loads(
                (result.run_dir / "run_provenance.json").read_text(encoding="utf-8")
            )
            if not provenance["train_preserved"]:
                raise ExportError(f"train changed for {campaign}")
            document = result.model_dump(mode="json")
            document["run_dir"] = str(final_group / campaign)
            document["detector_release_dir"] = str(final_group / campaign / "detector")
            document.update(
                {
                    "execution_sha256": execution_digest(summary),
                    "execution_reproducible": True,
                    "planned_events": len(summary.results),
                    "applied_events": len(summary.applied_results),
                    "rejected_events": len(summary.rejected_results),
                    "requested_quota": summary.requested_quota,
                    "reserved_capacity": summary.reserved_before,
                    "committed_budget": summary.committed_total,
                    "per_partition": provenance["per_partition_modifications"],
                    "train_preserved": True,
                    "per_family": provenance["per_family"],
                    "root_checksums_sha256": sha256_file(result.run_dir / "SHA256SUMS.txt"),
                }
            )
            evidence[campaign] = document
            print(
                f"{campaign}: {result.modified_rows} rows, {result.modified_cells} cells; "
                f"{len(summary.applied_results)}/{len(summary.results)} events; "
                f"quota deficit {result.quota_deficit}",
                flush=True,
            )
        pd.testing.assert_frame_equal(dataset.frame, snapshot, check_exact=True)
        verify_originals()
        write_json(
            staging / "export_summary.json",
            {
                "group_directory": str(final_group),
                "source_sha256": dataset.source_sha256,
                "protected_file_count": len(protected),
                "protected_files_unchanged": True,
                "detector_release_policy": "release only detector/; hold test truth until freeze",
                "campaigns": evidence,
            },
        )
        names = ["export_summary.json", *(f"{name}/SHA256SUMS.txt" for name in args.campaigns)]
        write_sha256sums(staging, hash_files(staging, names))

    group = run_in_temp_directory(args.output, group_id, write_group)
    verify_originals()
    print(f"Validated evaluator bundle: {group}")
    print(f"Summary: {group / 'export_summary.json'}")


if __name__ == "__main__":
    main()
