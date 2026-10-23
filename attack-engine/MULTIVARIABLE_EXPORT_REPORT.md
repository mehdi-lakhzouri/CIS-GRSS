# Multi-variable dataset export

Completed 23 October 2026. The existing Phase 5 pipeline now exports Spike, Bias, Drift,
Replay and Stealth across all selected variables. Five independent campaigns and a separate
combined campaign were generated, exported and validated against the real USCRN delivery.
The original source delivery and previous Phase 5 export were preserved.

**Release only a campaign's `detector/` subdirectory to the detector team. Its parent contains
evaluator-only ground truth, clean references and attack details, including test labels.**

## Export paths

All six exports are under this new directory:

`attack-engine/datasets/generated/multivariable-20261023-seed14-acb26f07/`

| Campaign | Evaluator bundle, relative to that directory | Detector release |
|---|---|---|
| Spike | `spike/` | `spike/detector/` |
| Bias | `bias/` | `bias/detector/` |
| Drift | `drift/` | `drift/detector/` |
| Replay | `replay/` | `replay/detector/` |
| Stealth | `stealth/` | `stealth/detector/` |
| All five combined | `combined/` | `combined/detector/` |

[Export index](datasets/generated/multivariable-20261023-seed14-acb26f07/export_summary.json)
contains absolute paths, event counts, row/cell counts, quotas, per-family statistics,
partition statistics and every artifact's SHA-256. A repository-visible copy with test
evidence is [multivariable_export_results.json](validation/multivariable_export_results.json).

The group contains 92 files: 15 per campaign, plus the group index and checksum inventory.
Every campaign directory contains:

```text
uscrn_attacked.parquet          # Evaluator: original columns plus attacked copies and labels
detector_input.parquet          # Observable copy, identical to detector release input
ground_truth.parquet            # Evaluator: complete row ground truth
ground_truth_cells.parquet      # Evaluator: complete selected-cell ground truth
attack_manifest.csv             # Evaluator: every planned event, including rejections
data_contract.json              # Schemas, units, feature allowlist and audience policy
run_provenance.json             # Evaluator: configuration, calibration and accounting
HANDOVER.md                     # Evaluator instructions and run statistics
SHA256SUMS.txt                  # All campaign artifacts, including detector release files
detector/
    detector_input.parquet
    row_labels.parquet          # Authorized train/validation labels only; no value columns
    cell_labels.parquet         # Authorized train/validation labels only; no value columns
    data_contract.json
    HANDOVER.md                 # Detector instructions; no private run statistics/configuration
    SHA256SUMS.txt
```

## What changed

The exporter still consumes an already-committed execution; it does not rerun attacks or
change their scheduling. New per-variable overlays reuse the existing builders and writing
architecture. Each selected channel's finalized values replace that channel in
`detector_input.parquet`, including selected variables that previously appeared as context.
An explicit allowlist excludes labels, clean-value copies, attacked-value duplicate columns
and the clean-derived `flag_incoherence_verticale` flag.

The original single-target audit, row-truth, manifest and detector-input schemas remain
compatible. Existing file names and primary-channel semantics are retained; cell truth and
the separate detector release are additive. Generalized exports use contract version `2.0`;
the default single-target contract retains `1.0`. The seed-14 regression still has **9,679
modifications, including 690 Stealth**.

Each three-variable export contains 210,384 source rows, 22 detector-input columns and
631,152 ground-truth cells. The audit retains all 23 original columns exactly and adds
three attacked columns plus row labels/partition. No original column is rewritten.

## Ground truth and evaluability

Cell truth covers **every observation and selected variable**, including genuine cells,
unchanged replay values, missing observations and all train rows. It is keyed by
`(station, time, column)` and contains clean/final values, exact `is_attacked`, family, attack
ID and `is_evaluable`. Clean/final values must both be finite and differ exactly to count
as changed. Unchanged cells have null attack fields; missing values remain missing.

Row truth is the union of changed selected cells. Version 2 retains `clean_target` and
`attacked_target` as the first selected target's compatibility view and adds:

- `clean_values`, `attacked_values`, aligned with `target_columns` in the contract;
- `changed_columns`, `evaluable_columns`, and the row's `modified_cells` count;
- `attack_ids` and `attack_types`, preserving all identities when disjoint cell events share
  a row. Scalar ID/type fields remain useful for a single event.

A row is evaluable when at least one selected variable has a finite original value. For this
delivery, 210,257 rows and 618,970 selected cells are evaluable. Missing selected cells are
5,401 at 5 cm, 5,335 at 10 cm and 1,446 rainfall cells. The 127 rows missing all three selected
variables remain non-evaluable. Legacy single-target evaluability remains primary-specific.

The manifest preserves its primary `modified_count` and adds selected/changed variable
lists, modified rows, modified cells, per-variable counts, reserved/committed budget and the
budget unit. Provenance includes the resolved attack configuration, both budget caps,
per-partition/per-family accounting, frozen primary calibration identity and the actual
train-only secondary coupling parameters captured during execution. Engine source-file
hashes and dependency versions identify the implementation used for every export.

## Real USCRN results

All runs use seed 14, the three targets, validation/test attack placement and cell budgets.
Train modifications are **zero in every campaign**. The five independent configurations
remain under `configs/campaigns/<family>/`; their generation statistics match the previously
validated campaigns exactly.

| Campaign | Planned | Applied | Rejected | Cell quota | Reserved cells | Modified rows | Modified cells | Quota deficit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spike | 308 | 149 | 159 | 3,108 | 3,090 | 266 | 798 | 2,310 |
| Bias | 37 | 19 | 18 | 3,108 | 3,030 | 634 | 1,902 | 1,206 |
| Drift | 24 | 8 | 16 | 3,108 | 2,883 | 264 | 792 | 2,316 |
| Replay | 46 | 15 | 31 | 3,108 | 3,006 | 475 | 1,057 | 2,051 |
| Stealth | 33 | 16 | 17 | 3,108 | 2,961 | 477 | 969 | 2,139 |
| Combined | 356 | 152 | 204 | 12,433 | 11,616 | 1,342 | 3,597 | 8,836 |

| Campaign | SOIL_MOISTURE_5 changes | SOIL_MOISTURE_10 changes | P_CALC changes |
|---|---:|---:|---:|
| Spike | 266 | 266 | 266 |
| Bias | 634 | 634 | 634 |
| Drift | 264 | 264 | 264 |
| Replay | 472 | 468 | 117 |
| Stealth | 477 | 461 | 31 |
| Combined | 1,338 | 1,329 | 930 |

Rows are a union, not cells divided by three. Missing observations and genuine source values
that already equal a replay candidate stay unchanged. Quota deficits include unreserved
capacity and execution rejections; export never invents changes to fill them.

### Combined campaign allocation

The new [combined scheduling configuration](configs/campaigns/combined/scheduling.yaml)
uses `attack_budget: 0.12`, a denominator of 103,611 finite scoped observations, and equal
20% family weights. Integer quotas use the existing largest-remainder allocator. Its cell cap
is 12,433 and row cap is 4,300; all five families select three variables. Shared non-overlapping
windows and independent cell-write validation prevent conflicting writes. Parameters are in
[combined/attacks.yaml](configs/campaigns/combined/attacks.yaml).

| Family within combined run | Allocated cells | Reserved cells | Applied/planned | Rows | Cells | Quota deficit |
|---|---:|---:|---:|---:|---:|---:|
| Spike | 2,486 | 2,478 | 117/248 | 213 | 638 | 1,848 |
| Bias | 2,487 | 2,409 | 9/29 | 272 | 816 | 1,671 |
| Drift | 2,487 | 2,058 | 7/16 | 340 | 1,020 | 1,467 |
| Replay | 2,487 | 2,388 | 13/38 | 388 | 860 | 1,627 |
| Stealth | 2,486 | 2,283 | 6/25 | 129 | 263 | 2,223 |

Combined validation changes 710 rows / 1,909 cells; test changes 632 rows / 1,688 cells.
All five families have applied events. This is an independent campaign, not a concatenation
of the five individual datasets.

## Test-ground-truth restriction

The complete run parent is evaluator-owned. Its `detector/` directory contains the only
shareable release. Default label files contain 157,824 train/validation observation labels
and 473,472 train/validation cell labels, without clean or attacked value columns. Test labels
are absent; test observations remain in detector input so predictions can be produced.

`ExportConfiguration` rejects attempts to include `test`, unknown or duplicated partitions
in detector-label releases. It also supports withholding all training labels. Private attack
manifests, resolved configuration, clean audit data and full provenance are never copied into
the detector release. Separate evaluator and detector handovers explain the boundary and
the requirement to freeze predictions before accessing test ground truth.

Directory separation controls what is distributed. The evaluator must retain custody of the
parent bundle; this local exporter is not an identity/ACL service and does not implement an
automatic prediction-freeze or test-label release service.

## Checksums and source integrity

Source Parquet SHA-256, unchanged:

`46b98115424bb03af28d31bcb96b68fedd537a609bf97d4199a653a376b98421`

| Campaign | `detector_input.parquet` SHA-256, also identical in `detector/` |
|---|---|
| Spike | `c281eef40d8d5d5feb72fdda76d81f38fc2ad3cc839cc9ee944ad53129da8125` |
| Bias | `a83c107e6c094db6d8b544692fc344e2726262b4006716477d002830fac5d4bc` |
| Drift | `0256ae9c4beb274da9c8f903a45040e566c91cca2e68d0964c02ecf49c614735` |
| Replay | `61f91330e394a2d1917e0ccfed68e2b7699cddc076617e85fde8275bca91af2a` |
| Stealth | `2ec5205ab0e7e5cb37322564babf57b6ff8f6e1c7d948d4953feb4646e31bd7e` |
| Combined | `ff478b38f2f0996b5301a48f93f0a4b0a40b7bc321440c69f8e85ff31c6eb50a` |

Group `export_summary.json` SHA-256:
`4f6941ecc0d5b0bfcdebdc8323b6df9329a1c0bae6754a1785cce0f492ab69b3`.

Group `SHA256SUMS.txt` SHA-256:
`f292a485edc686b33ba9664486518d3aef5dd4397368d639441dae181ceb0c8d`.

Each campaign's checksum inventory covers its datasets, manifest, contract, provenance,
handovers and detector release. Each detector release has its own inventory. The group
inventory covers the index and campaign checksum inventories. There is no circular
self-hashing. All inventories were independently reopened and verified after publication;
complete per-file hashes are also recorded in the linked validation metadata.

All **65 protected original files** still match
[the initial hash inventory](validation/multivariable_export_integrity_before.json), including
all eight files in the prior `run-20261022T211937Z-seed14-a81da0` export. For example, its
detector-input hash remains
`a6970ddb9ffe5e5247c57cd4d445df02c2e93a16d87be1662bf868e56207d089`.
No previous dataset or export was overwritten.

## Validation performed

- **544 tests passed**, zero skipped, with the real USCRN dataset enabled. This includes
  16 real-data cases and six new export integrations, one for every independent/combined
  campaign. Full-suite runtime: 357.74 seconds.
- Ruff lint and formatting checks passed. `git diff --check` passed.
- Every real campaign was exported twice in isolated test directories. All 12 deterministic
  data/manifest/contract/release artifact hashes matched across repetitions. Provenance and
  the private handover contain run-specific identities/timestamps and are not required to
  match byte-for-byte.
- The durable-export driver repeated generation and matched execution fingerprints. It
  matched the five individual campaigns' previously validated row/cell counts, verified train
  preservation and checked every protected original file before publishing the complete group.
- In-memory and read-back checks reconcile exact cell masks, row unions, per-variable and
  per-family counts, event identities, manifest commits, both budget policies/caps, source
  values, missingness, detector observations and train preservation. JSON metadata is also
  compared with the constructed contract/provenance before publication.
- Synthetic tests cover all families under both budget units, target subsets/non-default
  primary targets, zero budgets, missing cells, multiple disjoint cell events in a row,
  withheld training labels, test-label denial, in-memory reference mutation, corrupted writes,
  duplicate runs and publication races. Failed writes/validation remove only their private
  staging directories and publish no completed run.
- The baseline calibration and train-only/no-leakage generation tests remain passing. Every
  published campaign's recorded engine source hashes match the current implementation.

## Reproduction

From `attack-engine`, export all six campaigns into another uniquely named group:

```powershell
uv run python scripts/export_multivariable.py
```

Use `--campaigns spike` or another family to export it independently. An existing group ID
is rejected. For a single campaign through the existing CLI:

```powershell
uv run attack-engine export `
  --input ../USCRN_livraison/data/uscrn_clean.parquet `
  --config configs/dataset.yaml `
  --attack-config configs/campaigns/combined/scheduling.yaml `
  --attack-params configs/campaigns/combined/attacks.yaml `
  --output datasets/generated
```

Validation commands:

```powershell
$env:ATTACK_ENGINE_REAL_DATASET = (Resolve-Path ../USCRN_livraison/data/uscrn_clean.parquet).Path
uv run pytest -q --basetemp=.pytest-tmp-multivariable-export-review --tb=short
uv run ruff check .
uv run ruff format --check .
```

## Limitations and stopping point

Exports preserve the generator's configured physical constraints and conservative atomic
rejections; they do not improve attack feasibility or fill quotas. The combined campaign
uses shared per-variable limits suitable for the five strategies; it remains distinct from
the independently configured campaigns. Existing scheduler window overlap restrictions are
retained. Reproducibility assumes the recorded input, configuration, implementation and
dependency versions; metadata timestamps intentionally vary between runs.

The reference driver checks the captured USCRN/configuration integrity inventory and refuses
to treat edited reference inputs as the validated delivery. General programmatic/CLI export
remains available through the existing architecture.

Export and validation are complete. No AI benchmarking, ERA5 integration or Sensor Emulator
was implemented. Test ground truth remains withheld from detector releases.
