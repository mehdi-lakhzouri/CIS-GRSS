"""Generates ``HANDOVER.md``: the human-readable guide for the AI/evaluation team."""

from __future__ import annotations

from attack_engine.exporting.models import DatasetContract, RunProvenance


def build_handover_markdown(
    contract: DatasetContract, provenance: RunProvenance, result_row_count: int
) -> str:
    context_columns = ", ".join(f"`{name}`" for name in contract.detector_context_columns)
    excluded = "\n".join(
        f"- `{name}`: {reason}" for name, reason in contract.excluded_from_detector.items()
    )
    partitions = "\n".join(
        f"- `{p.name}`: {p.start.isoformat()} .. {p.end.isoformat()} (measured interval)"
        for p in provenance.partitions
    )
    label_partitions = ", ".join(f"`{name}`" for name in contract.detector_label_partitions)
    per_family = "\n".join(
        f"| {fam.family} | {fam.planned_windows} | {fam.applied_windows} | "
        f"{fam.rejected_windows} | {fam.committed} | {fam.deficit} |"
        for fam in provenance.per_family
    )
    files_table = "\n".join(
        [
            "| File | Audience | Contents |",
            "|---|---|---|",
            "| `uscrn_attacked.parquet` | Internal audit / evaluators only | All original "
            f"columns untouched, plus `{contract.attacked_target_column}`, `is_attacked`, "
            "`attack_type`, `attack_id`, `partition`. Not detector input. |",
            "| `detector_input.parquet` | AI detector team | Observation keys, approved "
            "temporal metadata, the observed (attacked) target, and an explicit allowlist "
            "of environmental context columns. No label or privileged column. |",
            "| `ground_truth.parquet` | Evaluators, and the AI team for authorized "
            "partitions only | One row per source observation: `clean_target`, "
            "`attacked_target`, `is_attacked`, `attack_type`, `attack_id`, `is_evaluable`. |",
            "| `attack_manifest.csv` | Evaluators only | One row per planned attack "
            "window: status, requested/reserved capacity, actual modified count, "
            "rejection reason, realized magnitude statistics. |",
            "| `run_provenance.json` | Evaluators, reproducibility audits | Input/config "
            "identities and hashes, seed, dependency versions, budget and per-family "
            "accounting, output file hashes. |",
            "| `data_contract.json` | Everyone | Machine-readable version of this "
            "document's schema decisions. |",
            "| `SHA256SUMS.txt` | Everyone | SHA-256 of every file in this directory, "
            "including `run_provenance.json` itself. |",
        ]
    )
    return f"""# Dataset handover - run `{provenance.run_id}`

## Purpose

This bundle is the Phase 5 export of an independent False Data Injection attack
engine (TSYP14, Track 2) built on cleaned USCRN hourly observations. It carries
injected attacks against `{contract.target_column}` from five families -
spike, bias, drift, replay and context-aware stealth - produced by the already
validated Phases 1-4 attack-generation pipeline. Nothing in this directory
re-runs or re-derives an attack; it only arranges already-committed results.

## Files and audience

{files_table}

## Schema and units

Observation keys are `(station, time)`; `time` is an interval-end UTC timestamp,
so the record stamped `2025-01-01T00:00:00Z` closes the last hour of 2024 and
belongs to the `test` partition. The target, `{contract.target_column}`, is
volumetric soil water content (m3/m3) at 5 cm depth.

## Partitions (exactly Phase 1's boundaries)

{partitions}

Train covers 2022-2023, validation covers 2024, test covers 2025.

## Nullable values

Missing values in the original delivery remain missing everywhere: a row whose
clean target was `NaN` keeps `clean_target`, `attacked_target` and detector
`{contract.target_column}` as `NaN`/null, `is_attacked = false`, and
`is_evaluable = false` (point-level attack detection is meaningless without an
original value to compare against). Missing contextual measurements are never
imputed.

## Detector feature allowlist

`detector_input.parquet` contains exactly: `station`, `time`, `partition`,
`{contract.target_column}` (the attacked, observed value), and:
{context_columns}.

Excluded, with reasons:
{excluded}

Inspect `data_contract.json` for the authoritative, versioned list.

## Label semantics

`is_attacked = 1` only if the final attacked target differs from a *finite*
original target; a zero-effect attack event or a row with a missing original
value is never labeled positive. Every positive label carries a non-null
`attack_id` traceable to exactly one applied event in `attack_manifest.csv`.

## Ground-truth access policy

The AI team may use ground-truth labels for `{label_partitions}` for training
and model selection. Test-partition labels are restricted to the evaluator
until predictions are frozen - do not share `ground_truth.parquet` rows for
`test` with the detector team before that point.

## Joining predictions with ground truth

The expected AI output is `predictions.parquet` keyed by `(station, time)`,
with columns `anomaly_score` (continuous) and `predicted_anomaly` (boolean or
0/1), plus optional `model_name`/`model_version` metadata. Join it to
`ground_truth.parquet` on `(station, time)` to compute detection metrics; this
phase does not implement that benchmark.

## Data leakage precautions

`detector_input.parquet` never carries `is_attacked`, `attack_type`,
`attack_id`, a second clean-target column, or `flag_incoherence_verticale`
(derived from the clean reference). Do not join `uscrn_attacked.parquet` or
`ground_truth.parquet` into a detector-facing pipeline.

## Reproducibility

```powershell
uv run attack-engine export `
  --input {provenance.input_path} `
  --config configs/dataset.yaml `
  --attack-config configs/scheduling.yaml `
  --attack-params configs/attacks.yaml `
  --output datasets/generated
```

Input SHA-256: `{provenance.input_sha256}`. Global seed: `{provenance.global_seed}`.
See `run_provenance.json` for the full dependency and configuration identity.

## Per-family summary (this run)

| Family | Planned | Applied | Rejected | Committed | Deficit |
|---|---:|---:|---:|---:|---:|
{per_family}

Total rows: {result_row_count}. Committed modifications: {provenance.committed_modifications}.

## Checksums and integrity

Verify every file against `SHA256SUMS.txt` (standard `sha256sum -c` format).
The source Parquet's SHA-256 was checked unchanged before and after this
export; see `run_provenance.json`'s `input_sha256`.

## Known limitations

This bundle documents one export run of an experimental attack generator; it
makes no claim about detector evasion, detection efficacy or online attack
success. See the project `README.md` and `PHASE4_REPORT.md` for the
underlying generation methodology and its limitations.
"""
