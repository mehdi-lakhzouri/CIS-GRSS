"""Scientific and consistency checks run before an export is declared valid.

Every check reads already-built DataFrames and upstream results; none of them
recompute an attack or mutate anything. A failing check raises
:class:`~attack_engine.exporting.models.ExportError` via :func:`require_valid`,
or is collected as a message via :func:`run_consistency_checks`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from attack_engine.attacks.executor import ExecutionSummary
from attack_engine.dataset.schema import DatasetSchema
from attack_engine.exporting.contracts import PRIVILEGED_COLUMNS
from attack_engine.exporting.models import ExportError


@dataclass(frozen=True)
class CheckResult:
    code: str
    passed: bool
    message: str


def run_consistency_checks(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    summary: ExecutionSummary,
    audit: pd.DataFrame,
    ground_truth: pd.DataFrame,
    detector_input: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    cell_ground_truth: pd.DataFrame | None = None,
    target_columns: tuple[str, ...] | None = None,
    plan=None,
    contract=None,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    row_count = len(frame)

    checks.append(_row_count_check("audit_row_count", audit, row_count))
    checks.append(_row_count_check("ground_truth_row_count", ground_truth, row_count))
    checks.append(_row_count_check("detector_input_row_count", detector_input, row_count))

    checks.append(_unique_keys_check("ground_truth_unique_keys", ground_truth))
    checks.append(_unique_keys_check("detector_input_unique_keys", detector_input))

    checks.append(_order_preserved_check(frame, schema, ground_truth))
    checks.append(_source_columns_preserved_check(frame, schema, audit))
    if cell_ground_truth is not None:
        checks.extend(
            _variable_checks(
                frame,
                schema,
                summary,
                audit,
                ground_truth,
                cell_ground_truth,
                detector_input,
                manifest,
                target_columns,
                plan,
                contract,
            )
        )
        return checks
    checks.append(_committed_differ_check(ground_truth))
    checks.append(_missing_target_check(ground_truth))
    checks.append(_label_traceability_check(ground_truth))
    checks.append(_manifest_reconciliation_check(manifest, ground_truth, summary))
    checks.append(_per_family_reconciliation_check(ground_truth, summary))
    checks.append(_no_overlap_check(ground_truth, summary))
    checks.append(_detector_input_no_leakage_check(detector_input))
    return checks


def require_valid(checks: list[CheckResult]) -> None:
    failed = [check for check in checks if not check.passed]
    if failed:
        details = "; ".join(f"{check.code}: {check.message}" for check in failed)
        raise ExportError(f"export failed {len(failed)} consistency check(s): {details}")


def _row_count_check(code: str, table: pd.DataFrame, expected: int) -> CheckResult:
    passed = len(table) == expected
    return CheckResult(code, passed, f"{len(table)} rows, expected {expected}")


def _unique_keys_check(code: str, table: pd.DataFrame) -> CheckResult:
    duplicates = int(table.duplicated(subset=["station", "time"]).sum())
    return CheckResult(code, duplicates == 0, f"{duplicates} duplicate (station, time) keys")


def _order_preserved_check(
    frame: pd.DataFrame, schema: DatasetSchema, ground_truth: pd.DataFrame
) -> CheckResult:
    original = frame[[schema.station_column, schema.timestamps.column]].to_numpy()
    exported = ground_truth[["station", "time"]].to_numpy()
    passed = bool((original == exported).all())
    message = (
        "observation keys preserved in original order"
        if passed
        else "ground truth keys do not match the original row order"
    )
    return CheckResult("row_order_preserved", passed, message)


def _source_columns_preserved_check(
    frame: pd.DataFrame, schema: DatasetSchema, audit: pd.DataFrame
) -> CheckResult:
    try:
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True)[list(schema.column_names)],
            audit[list(schema.column_names)],
            check_exact=True,
        )
    except AssertionError as exc:
        return CheckResult("source_columns_preserved", False, str(exc).splitlines()[0])
    return CheckResult("source_columns_preserved", True, "all source columns preserved exactly")


def _committed_differ_check(ground_truth: pd.DataFrame) -> CheckResult:
    attacked = ground_truth[ground_truth["is_attacked"]]
    if attacked.empty:
        return CheckResult("committed_positions_differ", True, "no attacked rows")
    passed = bool((attacked["attacked_target"] != attacked["clean_target"]).all())
    return CheckResult(
        "committed_positions_differ", passed, f"{len(attacked)} attacked rows checked"
    )


def _missing_target_check(ground_truth: pd.DataFrame) -> CheckResult:
    missing = ground_truth[ground_truth["clean_target"].isna()]
    if missing.empty:
        return CheckResult("missing_target_untouched", True, "no missing clean targets")
    passed = bool(
        missing["attacked_target"].isna().all()
        and not missing["is_attacked"].any()
        and not missing["is_evaluable"].any()
    )
    return CheckResult(
        "missing_target_untouched", passed, f"{len(missing)} missing-target rows checked"
    )


def _label_traceability_check(ground_truth: pd.DataFrame) -> CheckResult:
    attacked = ground_truth["is_attacked"]
    has_id = ground_truth["attack_id"].notna()
    missing_id = bool((attacked & ~has_id).any())
    stray_id = bool((~attacked & has_id).any())
    passed = not missing_id and not stray_id
    return CheckResult(
        "label_traceability",
        passed,
        f"missing_id={missing_id} stray_id={stray_id}",
    )


def _manifest_reconciliation_check(
    manifest: pd.DataFrame, ground_truth: pd.DataFrame, summary: ExecutionSummary
) -> CheckResult:
    manifest_committed = int(manifest.loc[manifest["status"] == "applied", "modified_count"].sum())
    ground_truth_committed = int(ground_truth["is_attacked"].sum())
    ledger_committed = summary.committed_total
    passed = manifest_committed == ground_truth_committed == ledger_committed
    return CheckResult(
        "manifest_reconciles_with_labels",
        passed,
        f"manifest={manifest_committed} ground_truth={ground_truth_committed} "
        f"ledger={ledger_committed}",
    )


def _per_family_reconciliation_check(
    ground_truth: pd.DataFrame, summary: ExecutionSummary
) -> CheckResult:
    counts = ground_truth.loc[ground_truth["is_attacked"], "attack_type"].value_counts()
    mismatches = {
        family.value: (counts.get(family.value, 0), execution.committed)
        for family, execution in summary.per_family.items()
        if counts.get(family.value, 0) != execution.committed
    }
    return CheckResult(
        "per_family_reconciles_with_executor", not mismatches, f"mismatches={mismatches}"
    )


def _no_overlap_check(ground_truth: pd.DataFrame, summary: ExecutionSummary) -> CheckResult:
    total_changed_mask = sum(result.modified_count for result in summary.applied_results)
    labeled = int(ground_truth["is_attacked"].sum())
    passed = total_changed_mask == labeled
    return CheckResult(
        "no_overlapping_modifications",
        passed,
        f"sum(result.modified_count)={total_changed_mask} labeled={labeled}",
    )


def _detector_input_no_leakage_check(detector_input: pd.DataFrame) -> CheckResult:
    leaked = set(detector_input.columns) & set(PRIVILEGED_COLUMNS)
    return CheckResult("detector_input_no_leakage", not leaked, f"leaked={sorted(leaked)}")


def _variable_checks(
    frame, schema, summary, audit, truth, cells, detector, manifest, targets, plan, contract
) -> list[CheckResult]:
    """Reconcile values, identities and budgets independently from the overlay builder."""
    checks = []

    def check(code, passed, message):
        checks.append(CheckResult(code, bool(passed), message))

    count = len(frame)
    check(
        "cell_ground_truth_count", len(cells) == count * len(targets), "full selected-cell coverage"
    )
    check(
        "cell_unique_keys",
        not cells.duplicated(["station", "time", "column"]).any(),
        "unique cell keys",
    )
    if len(cells) != count * len(targets):
        return checks
    masks, expected_written, observed_written = [], {}, {}
    for result in summary.applied_results:
        check(
            "primary_count_exact",
            result.modified_count == sum(result.changed_mask),
            result.window_id,
        )
        for column in result.columns:
            for label, value, changed in zip(
                column.row_labels, column.candidate_values, column.changed_mask, strict=True
            ):
                if changed:
                    key = (label, column.column)
                    check("no_overlapping_modifications", key not in expected_written, str(key))
                    expected_written[key] = (value, result.family.value, result.window_id)
    for name in targets:
        table = cells.loc[cells.column.eq(name)].reset_index(drop=True)
        if len(table) != count:
            check("cell_column_coverage", False, name)
            continue
        clean = frame[name].to_numpy(dtype=float)
        final = audit[f"{name}_attacked"].to_numpy(dtype=float)
        actual = np.isfinite(clean) & np.isfinite(final) & (clean != final)
        masks.append(actual)
        check(
            "cell_keys_order",
            np.array_equal(
                table[["station", "time"]].to_numpy(),
                frame[[schema.station_column, schema.timestamps.column]].to_numpy(),
            ),
            name,
        )
        check(
            "cell_partitions",
            np.array_equal(table.partition.to_numpy(), truth.partition.to_numpy()),
            name,
        )
        check("cell_clean_exact", np.array_equal(table.clean_value, clean, equal_nan=True), name)
        check("cell_final_exact", np.array_equal(table.attacked_value, final, equal_nan=True), name)
        check(
            "detector_observations_exact",
            name in detector and np.array_equal(detector[name], final, equal_nan=True),
            name,
        )
        check("exact_cell_masks", np.array_equal(table.is_attacked, actual), name)
        check("cell_evaluability", np.array_equal(table.is_evaluable, np.isfinite(clean)), name)
        check(
            "missing_target_untouched",
            np.array_equal(np.isnan(clean), np.isnan(final))
            and not np.any(np.isfinite(clean) & ~np.isfinite(final)),
            name,
        )
        check(
            "cell_label_traceability",
            np.array_equal(table.attack_id.notna(), actual)
            and np.array_equal(table.attack_type.notna(), actual),
            name,
        )
        for pos in np.flatnonzero(actual):
            observed_written[(frame.index[pos], name)] = (
                final[pos],
                table.attack_type.iloc[pos],
                table.attack_id.iloc[pos],
            )
    if len(masks) != len(targets):
        return checks
    matrix = np.column_stack(masks)
    row_mask = matrix.any(axis=1)
    clean_matrix = frame.loc[:, targets].to_numpy(dtype=float)
    check("row_union_exact", np.array_equal(truth.is_attacked, row_mask), "row label is cell union")
    check(
        "row_evaluability",
        np.array_equal(truth.is_evaluable, np.isfinite(clean_matrix).any(axis=1)),
        "at least one finite target",
    )
    if "changed_columns" in truth:
        expected_columns = [
            [c for c, flag in zip(targets, row, strict=True) if flag] for row in matrix
        ]
        check(
            "changed_columns_exact",
            truth.changed_columns.tolist() == expected_columns,
            "per-row changed variables",
        )
        check(
            "row_cell_counts",
            np.array_equal(truth.modified_cells, matrix.sum(axis=1)),
            "per-row modified-cell counts",
        )
        check(
            "row_clean_values",
            np.array_equal(np.asarray(truth.clean_values.tolist()), clean_matrix, equal_nan=True),
            "aligned target values",
        )
        finals = np.column_stack([audit[f"{c}_attacked"] for c in targets])
        check(
            "row_final_values",
            np.array_equal(np.asarray(truth.attacked_values.tolist()), finals, equal_nan=True),
            "aligned target values",
        )
        expected_ids, expected_types = {}, {}
        for (label, _), (_, family, event_id) in expected_written.items():
            expected_ids.setdefault(label, set()).add(event_id)
            expected_types.setdefault(label, set()).add(family)
        check(
            "row_event_identity",
            truth.attack_ids.tolist()
            == [sorted(expected_ids.get(label, set())) for label in frame.index]
            and truth.attack_types.tolist()
            == [sorted(expected_types.get(label, set())) for label in frame.index],
            "all row event identities",
        )
    check(
        "exact_execution_overlay",
        observed_written == expected_written,
        "cell values, families and IDs match committed results",
    )
    check("row_accounting", int(row_mask.sum()) == summary.committed_rows, "modified row union")
    check("cell_accounting", int(matrix.sum()) == summary.committed_cells, "modified cell total")
    counts = {name: int(matrix[:, i].sum()) for i, name in enumerate(targets)}
    check(
        "per_variable_accounting",
        counts == {c: summary.per_variable_modifications.get(c, 0) for c in targets},
        str(counts),
    )
    unit_count = (
        summary.committed_cells if summary.budget_unit == "cells" else summary.committed_rows
    )
    check(
        "budget_accounting",
        summary.committed_total == unit_count == plan.ledger.total_committed
        and unit_count <= summary.reserved_before <= summary.requested_quota,
        "explicit row/cell budget",
    )
    check(
        "row_cap",
        plan.max_modified_rows is None or summary.committed_rows <= plan.max_modified_rows,
        "row cap respected",
    )
    check(
        "cell_cap",
        plan.max_modified_cells is None or summary.committed_cells <= plan.max_modified_cells,
        "cell cap respected",
    )
    check(
        "train_preservation",
        "train" in plan.scope or not row_mask[truth.partition.eq("train")].any(),
        "train genuine when outside attack scope",
    )
    by_id = {r.window_id: r for r in summary.results}
    windows_by_id = {w.window_id: w for w in plan.windows}
    check(
        "manifest_plan_identity",
        manifest.attack_id.tolist() == [w.window_id for w in plan.windows],
        "one manifest record per planned event",
    )
    for record in manifest.to_dict("records"):
        result = by_id.get(record["attack_id"])
        window = windows_by_id.get(record["attack_id"])
        if window is None:
            check("manifest_event_identity", False, record["attack_id"])
            continue
        applied = result is not None and result.applied
        check("manifest_status", (record["status"] == "applied") == applied, record["attack_id"])
        check(
            "manifest_event_identity",
            record["family"] == window.family.value
            and record["station"] == window.station
            and record["partition"] == window.partition,
            record["attack_id"],
        )
        check(
            "manifest_primary_count",
            record["modified_count"] == (result.modified_count if applied else 0),
            record["attack_id"],
        )
        if "modified_cells" in record:
            check(
                "manifest_variable_counts",
                json.loads(record["per_variable_modifications"])
                == (result.per_variable_modifications if result else {})
                and json.loads(record["changed_variables"])
                == list(result.changed_variables if result else ()),
                record["attack_id"],
            )
            check(
                "manifest_row_cell_counts",
                record["modified_cells"] == (result.modified_cells if applied else 0)
                and record["modified_rows"] == (result.modified_rows if applied else 0),
                record["attack_id"],
            )
            check(
                "manifest_budget",
                record["committed_budget"]
                == (plan.ledger.reservation(record["attack_id"]).committed_count or 0),
                record["attack_id"],
            )
    for family, execution in summary.per_family.items():
        table = cells.loc[cells.is_attacked & cells.attack_type.eq(family.value)]
        family_rows = len(table[["station", "time"]].drop_duplicates())
        family_cells = len(table)
        check(
            "per_family_reconciles_with_executor",
            family_cells == execution.modified_cells
            and family_rows == execution.modified_rows
            and execution.committed
            == (family_cells if summary.budget_unit == "cells" else family_rows),
            family.value,
        )
    check(
        "detector_input_no_leakage",
        tuple(detector.columns) == contract.detector_input_columns
        and not (set(detector.columns) & set(PRIVILEGED_COLUMNS)),
        "exact observable feature allowlist",
    )
    for name in contract.detector_context_columns:
        if name not in targets:
            check(
                "context_values_exact",
                np.array_equal(detector[name].to_numpy(), frame[name].to_numpy(), equal_nan=True),
                name,
            )
    return checks
