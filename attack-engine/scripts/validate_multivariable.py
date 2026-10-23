"""Read-only USCRN campaign validation. Writes statistics and integrity evidence only.

Run from attack-engine: uv run python scripts/validate_multivariable.py
No observation datasets, calibration artifacts or previous exports are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from attack_engine.attacks.config import load_attack_config
from attack_engine.attacks.executor import AttackExecutor
from attack_engine.attacks.multistealth import fit_secondary_calibration
from attack_engine.config import load_config
from attack_engine.dataset.loader import DatasetLoader
from attack_engine.dataset.schema import uscrn_hourly_schema
from attack_engine.dataset.temporal import assign_partitions, to_utc
from attack_engine.scheduling.config import load_scheduling_config
from attack_engine.scheduling.models import AttackFamily
from attack_engine.scheduling.scheduler import AttackWindowScheduler
from attack_engine.utils.hashing import sha256_file

PROJECT = Path(__file__).resolve().parents[1]
VARIABLES = ("SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC")


def execution_digest(summary):
    """Fingerprint event identity, statuses, all emitted arrays and exact masks."""
    digest = hashlib.sha256()
    for result in summary.results:
        digest.update(
            json.dumps(
                [
                    result.window_id,
                    result.family.value,
                    result.status.value,
                    result.rejection_reason,
                    result.detail,
                    result.resolved_target_columns,
                ]
            ).encode()
        )
        for column in result.columns:
            digest.update(column.column.encode())
            digest.update(np.asarray(column.row_labels, dtype="<i8").tobytes())
            digest.update(np.asarray(column.candidate_values, dtype="<f8").tobytes())
            digest.update(np.asarray(column.changed_mask, dtype="bool").tobytes())
    return digest.hexdigest()


def audit(frame, partitions, scheduling, config, plan, summary):
    """Reconstruct emissions independently and reconcile every cell and budget unit."""
    observed = frame.loc[:, VARIABLES].copy(deep=True)
    written, rows, per_variable = set(), set(), Counter()
    times = to_utc(frame.time)
    partition = assign_partitions(times, partitions, pd.Timedelta(hours=1))
    for result in summary.results:
        assert result.partition in scheduling.budget_scope
        assert result.changed_variables == (
            result.resolved_target_columns if result.applied else ()
        )
        for column in result.columns:
            clean, final = np.asarray(column.clean_values), np.asarray(column.candidate_values)
            np.testing.assert_array_equal(clean, frame.loc[list(column.row_labels), column.column])
            mask = np.isfinite(clean) & np.isfinite(final) & (clean != final)
            np.testing.assert_array_equal(mask, column.changed_mask)
            np.testing.assert_array_equal(np.isnan(clean), np.isnan(final))
            assert int(mask.sum()) == column.modified_count
            assert np.all(final[mask] >= 0)
            assert np.all(final[mask] <= (200 if column.column == "P_CALC" else 0.7))
            per_variable[column.column] += column.modified_count
            for label, value, changed in zip(column.row_labels, final, mask, strict=True):
                if changed:
                    assert (label, column.column) not in written
                    written.add((label, column.column))
                    rows.add(label)
                    observed.at[label, column.column] = value
            if result.applied and result.family is AttackFamily.REPLAY:
                source = result.replay_source
                assert source.last_time < times.at[result.row_labels[0]]
                assert set(frame.loc[list(source.row_labels), "station"]) == {result.station}
                assert set(partition.loc[list(source.row_labels)]) == {result.partition}
                genuine = frame.loc[list(source.row_labels), column.column].to_numpy()
                np.testing.assert_array_equal(
                    final[np.isfinite(clean)], genuine[np.isfinite(clean)]
                )
        reservation = plan.ledger.reservation(result.window_id)
        if result.applied:
            expected = (
                result.modified_cells if plan.budget_unit == "cells" else result.modified_rows
            )
            assert reservation.committed_count == expected
        else:
            assert result.modified_cells == 0
    assert len(written) == summary.committed_cells == sum(per_variable.values())
    assert len(rows) == summary.committed_rows
    assert dict(per_variable) == summary.per_variable_modifications
    assert summary.committed_total == plan.ledger.total_committed
    assert summary.committed_total == sum(f.committed for f in summary.per_family.values())
    assert summary.committed_total <= summary.reserved_before <= summary.requested_quota
    assert summary.committed_rows <= scheduling.max_modified_rows
    assert summary.committed_cells <= scheduling.max_modified_cells
    assert all(c.active_reservation_count == 0 for c in plan.ledger.cell_ledgers())
    # Independent final-frame checks cover adjoining events as well as clean boundaries.
    masks = (
        np.isfinite(observed)
        & np.isfinite(frame.loc[:, VARIABLES])
        & observed.ne(frame.loc[:, VARIABLES])
    )
    np.testing.assert_array_equal(
        observed.loc[partition.eq("train")], frame.loc[partition.eq("train"), VARIABLES]
    )
    assert int(masks.to_numpy().sum()) == summary.committed_cells
    ordered = frame.assign(_time=times, _partition=partition).sort_values(["station", "_time"])
    labels = ordered.index
    contiguous = (
        ordered.station.eq(ordered.station.shift())
        & ordered._partition.eq(ordered._partition.shift())
        & ordered._time.diff().eq(pd.Timedelta(hours=1))
    )
    for column, limits in config.variable_constraints.items():
        touched = masks.loc[labels, column]
        edges = contiguous & (touched | touched.shift(fill_value=False))
        values = observed.loc[labels, column]
        if limits.max_step_per_hour is not None:
            assert not (edges & values.diff().abs().gt(limits.max_step_per_hour + 1e-12)).any()
        if limits.max_offset_rate_per_hour is not None:
            offsets = values - frame.loc[labels, column]
            assert not (
                edges & offsets.diff().abs().gt(limits.max_offset_rate_per_hour + 1e-12)
            ).any()
    for relationship in config.relationships:
        left, right = relationship.left, relationship.right
        a, b = observed[left], observed[right]
        if relationship.quantity == "offsets":
            a, b = a - frame[left], b - frame[right]
        checked = (masks[left] | masks[right]) & np.isfinite(a) & np.isfinite(b)
        value = a - relationship.scale * b
        assert value[checked].between(relationship.lower - 1e-12, relationship.upper + 1e-12).all()


def statistics(summary, plan):
    return {
        "budget_unit": summary.budget_unit,
        "finite_denominator": plan.finite_denominator,
        "requested_quota": summary.requested_quota,
        "reserved": summary.reserved_before,
        "committed": summary.committed_total,
        "quota_deficit": summary.deficit_vs_quota,
        "planned_events": len(summary.results),
        "applied_events": len(summary.applied_results),
        "rejected_events": len(summary.rejected_results),
        "modified_rows": summary.committed_rows,
        "modified_cells": summary.committed_cells,
        "per_variable_modifications": summary.per_variable_modifications,
        "rejection_reasons": dict(
            Counter(r.rejection_reason.value for r in summary.rejected_results)
        ),
        "row_cap": plan.max_modified_rows,
        "cell_cap": plan.max_modified_cells,
        "frame_unchanged": summary.frame_unchanged,
        "execution_sha256": execution_digest(summary),
        "events": [
            {
                "window_id": r.window_id,
                "status": r.status.value,
                "rejection_reason": r.rejection_reason,
                "detail": r.detail,
                "changed_variables": r.changed_variables,
                "modified_rows": r.modified_rows,
                "modified_cells": r.modified_cells,
                "per_variable_modifications": r.per_variable_modifications,
            }
            for r in summary.results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT.parent / "USCRN_livraison/data/uscrn_clean.parquet"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "validation/multivariable_all_families_results.json",
    )
    args = parser.parse_args()
    dataset_config = load_config(PROJECT / "configs/dataset.yaml")
    schema = uscrn_hourly_schema()
    integrity_path = PROJECT / "validation/multivariable_integrity_before.json"
    protected = json.loads(integrity_path.read_text(encoding="utf-8"))

    def verify_originals():
        assert all(
            sha256_file(PROJECT.parent / path) == digest for path, digest in protected.items()
        )

    verify_originals()
    dataset = DatasetLoader(schema).load(args.input)
    assert dataset.source_sha256 == dataset_config.expected_sha256
    frame = dataset.frame
    snapshot = frame.copy(deep=True)
    baseline_config = load_attack_config(PROJECT / "configs/attacks.yaml")
    baseline_scheduling = load_scheduling_config(PROJECT / "configs/scheduling.yaml")
    baseline_plan = AttackWindowScheduler(
        schema, dataset_config.partitions, baseline_scheduling
    ).plan(frame)
    baseline = AttackExecutor(schema, dataset_config.partitions, baseline_config).execute(
        frame, baseline_plan
    )
    baseline_counts = {f.value: s.committed for f, s in baseline.per_family.items()}
    assert baseline_counts == {
        "spike": 936,
        "bias": 2798,
        "drift": 2839,
        "replay": 2416,
        "stealth": 690,
    }
    assert baseline.committed_total == baseline.committed_cells == baseline.committed_rows == 9679
    results = {}
    for family in AttackFamily:
        root = PROJECT / "configs/campaigns" / family.value
        scheduling = load_scheduling_config(root / "scheduling.yaml")
        config = load_attack_config(root / "attacks.yaml")
        scheduler = AttackWindowScheduler(schema, dataset_config.partitions, scheduling)
        executor = AttackExecutor(schema, dataset_config.partitions, config)
        plan = scheduler.plan(frame)
        first = executor.execute(frame, plan)
        second = executor.execute(frame, scheduler.plan(frame))
        assert execution_digest(first) == execution_digest(second)
        assert first.per_variable_modifications == second.per_variable_modifications
        assert first.applied_results
        audit(frame, dataset_config.partitions, scheduling, config, plan, first)
        result = statistics(first, plan)
        result["reproducible"] = True
        result["config_sha256"] = {
            name: sha256_file(root / name) for name in ("scheduling.yaml", "attacks.yaml")
        }
        results[family.value] = result
        print(
            f"{family.value}: {result['applied_events']}/{result['planned_events']} events, "
            f"{first.committed_rows} rows, {first.committed_cells} cells, "
            f"deficit {first.deficit_vs_quota}",
            flush=True,
        )
    stealth = load_attack_config(PROJECT / "configs/campaigns/stealth/attacks.yaml").stealth
    calibrated = fit_secondary_calibration(frame, schema, dataset_config.partitions, stealth)
    poisoned = frame.copy(deep=True)
    train = assign_partitions(
        to_utc(frame.time), dataset_config.partitions, schema.timestamps.interval
    ).eq("train")
    poisoned.loc[~train, list(VARIABLES)] = -99999.0
    assert calibrated == fit_secondary_calibration(
        poisoned, schema, dataset_config.partitions, stealth
    )
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)
    verify_originals()
    document = {
        "source_sha256": dataset.source_sha256,
        "protected_file_count": len(protected),
        "protected_files_unchanged": True,
        "secondary_calibration_train_only": True,
        "baseline": {"modified_cells": baseline.committed_cells, "per_family": baseline_counts},
        "campaigns": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Validation evidence: {args.output}")


if __name__ == "__main__":
    main()
