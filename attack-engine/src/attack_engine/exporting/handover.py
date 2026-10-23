"""Evaluator instructions and the separately shareable detector handover."""

from __future__ import annotations

from attack_engine.exporting.models import DatasetContract, RunProvenance


def build_detector_handover(contract: DatasetContract) -> str:
    targets = ", ".join(f"`{c}` ({contract.target_units[c]})" for c in contract.target_columns)
    features = ", ".join(f"`{c}`" for c in contract.detector_input_columns)
    authorized = ", ".join(contract.detector_label_partitions) or "none"
    return f"""# Detector release

Share this directory only. Its parent is the private evaluator bundle.

`detector_input.parquet` contains the observed values for all selected targets:
{targets}. Every selected channel is replaced by its finalized observation; no
second clean copy or attack label is included in the input file.

Exact observable columns: {features}.

`row_labels.parquet` and `cell_labels.parquet` contain labels for authorized
partitions only: {authorized}. They contain no clean or attacked value columns.
Cell labels include negative and non-evaluable selected cells. A cell is evaluable
iff its original value is finite; row evaluability means at least one selected
variable is finite. An attack label is true only for an exact finite change.

Test ground truth is restricted to the evaluator until AI predictions are frozen.
No test-partition labels, clean references, attack manifest, attack configuration
or evaluator provenance are included in this release. Do not copy those files
from the parent directory into a detector-facing pipeline.

Times are interval-end UTC timestamps. Partition membership follows the measured
hour; `2025-01-01T00:00:00Z` closes 2024 and belongs to validation, not test.
Missing observations remain missing and are never imputed by this export.

Freeze predictions before requesting evaluator access to test labels. Row
predictions use `(station, time)` keys; variable predictions additionally use
`column`. Prediction storage and AI benchmarking are outside this phase.

`data_contract.json` defines the schemas, units and audience policy.
Verify the files in this directory against its own `SHA256SUMS.txt`.
"""


def build_handover_markdown(
    contract: DatasetContract, provenance: RunProvenance, result_row_count: int
) -> str:
    targets = ", ".join(f"`{c}` ({contract.target_units[c]})" for c in contract.target_columns)
    boundaries = "\n".join(
        f"- {p.name}: {p.start.isoformat()} .. {p.end.isoformat()} (measured interval)"
        for p in provenance.partitions
    )
    families = "\n".join(
        f"| {p.family} | {p.planned_windows} | {p.applied_windows} | {p.rejected_windows} | "
        f"{p.modified_rows} | {p.modified_cells} | {p.committed} | {p.quota_deficit} |"
        for p in provenance.per_family
    )
    vector_note = (
        "`clean_values` and `attacked_values` are aligned with `target_columns` in the contract. "
        "`changed_columns`, `modified_cells`, `attack_ids` and `attack_types` describe the row "
        "union. Scalar attack fields describe a single event; use the arrays and cell table "
        "if several disjoint events affect one row."
        if contract.schema_version == "2.0"
        else "The original single-target row schema and primary count semantics are preserved."
    )
    return f"""# Evaluator handover - run `{provenance.run_id}`

This directory is the private evaluator bundle. Release only its `detector/`
subdirectory to the AI detector team. File separation implements the distribution
policy; the evaluator must retain filesystem custody of this parent directory.

## Files and audience

| File | Audience | Contents |
|---|---|---|
| `detector/` | Detector team | Observations, authorized labels and public instructions. |
| `detector_input.parquet` | Observable copy | Same observations as detector release. |
| `uscrn_attacked.parquet` | Evaluators | Clean columns, attacked copies and labels. |
| `ground_truth.parquet` | Evaluators only | Full row truth, including restricted test labels. |
| `ground_truth_cells.parquet` | Evaluators | All selected cells, negatives and missing. |
| `attack_manifest.csv` | Evaluators | Events, identities, counts and rejection details. |
| `run_provenance.json` | Evaluators | Input/config/calibration identity, counts and hashes. |
| `data_contract.json` | Everyone | Versioned table schemas, target units and audience policy. |
| `SHA256SUMS.txt` | Evaluators | All artifact hashes, excluding itself. |

## Targets and exact ground truth

Selected targets: {targets}.

Every cell label is derived independently from the finite clean and final values.
Missing values remain missing, with is_attacked=false and is_evaluable=false.
Row is_attacked is the union across selected variables. Row evaluability is
{contract.row_evaluability}. Each changed cell carries the exact family and attack
ID of its applied manifest event; untouched cells have null attack fields.

{vector_note}

Test-partition ground truth remains evaluator-only until predictions are frozen.
The detector release physically omits test labels and all clean-value columns.
Only authorized training/validation label files may be shared before freezing.
There is no automatic release of test labels or prediction-freeze service here.

## Partitions

{boundaries}

UTC timestamps close the measured hour: `2025-01-01T00:00:00Z` belongs to validation.
Train is preserved when excluded from the configured attack scope. For this run,
train_preserved={provenance.train_preserved}.

## Accounting

| Family | Planned | Applied | Rejected | Rows | Cells | Budget committed | Quota deficit |
|---|---:|---:|---:|---:|---:|---:|---:|
{families}

Source rows: {result_row_count}. Modified rows: {provenance.modified_rows}.
Modified cells: {provenance.modified_cells}. Budget unit: {provenance.budget_unit}.
Requested quota: {provenance.requested_quota}; quota deficit: {provenance.unfulfilled_budget}.
Per-variable counts are recorded in provenance and in the multi-variable manifest.
`modified_count` in the manifest retains the event primary-channel meaning.

## Reproduction and integrity

Input SHA-256: `{provenance.input_sha256}`. Seed: {provenance.global_seed}.

```powershell
uv run attack-engine export `
  --input "{provenance.input_path}" `
  --config configs/dataset.yaml `
  --attack-config "{provenance.scheduling_config_path}" `
  --attack-params "{provenance.attack_params_path}" `
  --output datasets/generated
```

Use a new run ID or the generated default; an existing run is never overwritten.
The recorded resolved configuration, calibration, dependencies and checksums permit
reproduction. Dataset and manifest bytes are deterministic for fixed inputs;
provenance timestamps and run identities are intentionally run-specific.

Verify all artifacts against `SHA256SUMS.txt`; verify the shareable directory
against its separate checksum inventory. Source/config/calibration hashes and
value/mask/accounting checks are verified before publishing the completed run.

## Scope and limitations

This export arranges already-committed FDI results and does not rerun attacks.
It implements no AI benchmark, ERA5 integration or Sensor Emulator. Physical
constraints and attack feasibility remain the generator's configured experiment
policies; no detector-evasion claim is made. Directory separation is a distribution
boundary, not a role-authentication service: do not share the evaluator parent.
"""
