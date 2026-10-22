"""Scientific and consistency checks run before an export is declared valid.

Every check reads already-built DataFrames and upstream results; none of them
recompute an attack or mutate anything. A failing check raises
:class:`~attack_engine.exporting.models.ExportError` via :func:`require_valid`,
or is collected as a message via :func:`run_consistency_checks`.
"""

from __future__ import annotations

from dataclasses import dataclass

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
