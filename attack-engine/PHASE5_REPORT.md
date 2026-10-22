# Phase 5 — Ground Truth, Provenance and Dataset Export

Verified 22 October 2026. Phase 5 completed for the original `attack-engine` project.
Development and inspection cover this workspace and its local documents only; no other
Attack Engine repository was consulted for this phase.

## Scope

Phase 5 is downstream of the already-validated Phases 1-4 pipeline: it reads one
`AttackExecutor.execute()` result and writes a versioned export bundle. It does not
re-run, recompute or re-plan an attack, and it does not implement AI models, sensor
emulation, cryptography, APIs or dashboards.

## Files added

| File | Responsibility |
|---|---|
| [models.py](src/attack_engine/exporting/models.py) | `ExportConfiguration`, `DatasetContract`, `GroundTruthRecord`, `AttackManifestRecord`, `RunProvenance`, `ExportRun`, `ExportResult` |
| [contracts.py](src/attack_engine/exporting/contracts.py) | Explicit detector-input allowlist and privileged-column list; `DatasetContract` builder |
| [builder.py](src/attack_engine/exporting/builder.py) | Attacked target overlay (changed-mask only) and the three exported DataFrames |
| [manifest.py](src/attack_engine/exporting/manifest.py) | Event-level manifest rows, one per planned window, from the scheduler's plan and the executor's results |
| [validation.py](src/attack_engine/exporting/validation.py) | The Phase 5 scientific/consistency checks, run before any file is written |
| [provenance.py](src/attack_engine/exporting/provenance.py) | `run_provenance.json` builder (file/config hashes, seed, dependency versions, budget accounting) |
| [handover.py](src/attack_engine/exporting/handover.py) | `HANDOVER.md` generation |
| [writer.py](src/attack_engine/exporting/writer.py) | Atomic temp-directory-then-rename writing, `SHA256SUMS.txt` |
| [pipeline.py](src/attack_engine/exporting/pipeline.py) | `export_run()`: the single orchestration entry point, including read-back verification |
| [cli.py](src/attack_engine/cli.py) | New `export` command, wired to the existing `validate` → `plan` → `generate` → `export` flow |

Tests added: [contracts](tests/test_exporting_contracts.py), [builder](tests/test_exporting_builder.py),
[validation](tests/test_exporting_validation.py), [pipeline](tests/test_exporting_pipeline.py),
[CLI](tests/test_cli_export.py), plus one opt-in real-delivery export check appended to
[test_real_dataset.py](tests/test_real_dataset.py). No existing module, test, configuration or
calibration artifact was modified; Phases 1-4 behavior is unchanged and their existing test
suite continues to pass unmodified.

## Export architecture

`export_run()` (in `pipeline.py`) is the single entry point:

1. Build the attacked target by overlaying only `applied` results' `changed_mask` positions
   onto a copy of the clean target column (`builder.build_attacked_target`). A zero-effect
   attack (applied, but `changed_mask` all `False`) contributes nothing; a rejected event is
   never read. Overlapping writes raise `ExportError` (the executor already prevents this;
   the check is a second, independent guard at the export boundary).
2. Assign partitions with the exact Phase 1 function (`temporal.assign_partitions`), so labels
   can never diverge from the scheduler's own partitioning.
3. Build `uscrn_attacked.parquet` (all 23 original columns untouched, plus
   `SOIL_MOISTURE_5_attacked`, `is_attacked`, `attack_type`, `attack_id`, `partition`),
   `ground_truth.parquet` (one row per observation, labels re-derived independently from the
   final values and cross-checked against the changed mask) and `detector_input.parquet`
   (keys, partition, the attacked value as the observable target, and an explicit
   environmental-context allowlist).
4. Build `attack_manifest.csv` from the scheduler's full plan (every window, win or lose) and
   the executor's results, so rejected and skipped windows are represented explicitly rather
   than omitted.
5. Run `validation.run_consistency_checks()` and refuse to write anything if a check fails.
6. Write everything to a private `.tmp-<run-id>-<random>` directory, hash the five data files,
   build `run_provenance.json` from those hashes (never its own), render `HANDOVER.md`, then
   write `SHA256SUMS.txt` over all seven files including `run_provenance.json`. Only then is
   the temporary directory renamed to its final `<run-id>` name; any exception discards the
   temporary directory entirely, and an existing final directory is never overwritten.
7. Independently reopen every written file (`pq.read_table`, `pd.read_csv`, `json.loads`) and
   recompute its SHA-256 to verify the files on disk actually match what was written.
8. Re-hash the source Parquet (and the Stealth calibration artifact, if used) and compare
   against the hash captured at load time, both before and after the heavy work.

## Dataset schema decisions

- **Detector input allowlist** (`contracts.default_detector_context_columns`): every schema
  column except the keys, the target itself, the `annee` year column (superseded by the
  explicit `partition` column) and the privileged columns below. It is a fixed, reviewable
  tuple, never "every available DataFrame column".
- **Privileged / excluded-from-detector columns**: `is_attacked`, `attack_type`, `attack_id`
  (ground-truth labels) and `flag_incoherence_verticale` (derived from the clean reference,
  so it could leak attack labels). `resolve_detector_context_columns` raises if a caller tries
  to override the allowlist with any of these, or with the target column itself.
- **Ground truth labels**: `is_attacked = 1` iff the finite clean target differs from the final
  attacked value; `is_evaluable = clean_target.notna()`. Both are computed directly from the
  final values (not from reserved positions), then cross-checked against the executor's own
  changed mask — a mismatch aborts the export rather than silently exporting one or the other.
- **Manifest status model**: `applied` / `rejected` / `skipped` (for a family this phase does
  not execute — none, today, since all five families are executable) / `cancelled` (reserved
  for a status this phase never produces, kept so the model stays stable).
- **`run_provenance.json`** never hashes itself; `output_files` covers the five primary data
  files (the datasets, the manifest and the data contract). `SHA256SUMS.txt` is written last
  and is the file that actually covers the provenance document and `HANDOVER.md`.

## Tests executed and results

```powershell
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

- `uv run ruff check .`: **all checks passed**.
- `uv run ruff format --check .`: **75 files already formatted**.
- Full suite without the real dataset: **387 passed, 4 skipped** (the four real-dataset tests
  that opt in via `ATTACK_ENGINE_REAL_DATASET`).
- Full suite with `ATTACK_ENGINE_REAL_DATASET` set to the real delivery: **392 passed, 0
  skipped** (5 real-dataset tests pass, including the new Phase 5 export check).

## Actual real-dataset export statistics

Reproduction command:

```powershell
uv run attack-engine export `
  --input ../USCRN_livraison/data/uscrn_clean.parquet `
  --config configs/dataset.yaml `
  --attack-config configs/scheduling.yaml `
  --attack-params configs/attacks.yaml `
  --output datasets/generated
```

Run directory: `datasets/generated/run-20261022T211937Z-seed14-a81da0/`.

- Input: 210,384 rows, 23 columns, SHA-256 `46b98115424bb03af28d31bcb96b68fedd537a609bf97d4199a653a376b98421` (matched the configured expectation, and was unchanged after export).
- Exported row count: 210,384 in all three datasets (`uscrn_attacked.parquet`,
  `detector_input.parquet`, `ground_truth.parquet`), with unique `(station, time)` keys.
- Finite target: 204,983; missing target preserved: **5,401** (matches Phase 1-4).
- Committed modifications: **9,679**, exactly matching the PHASE4_REPORT.md reference run.
- Manifest events: 770 (534 spike + 70 bias + 42 drift + 75 replay + 49 stealth planned
  windows), `modified_count` summed over `status == applied` rows also equals 9,679.
- `Validation: PASSED` (every consistency check in `validation.py` passed).

## Per-family exported attack counts

| Family | Committed (ground truth / manifest, reconciled) |
|---|---:|
| spike | 936 |
| bias | 2,798 |
| drift | 2,839 |
| replay | 2,416 |
| stealth | 690 |
| **Total** | **9,679** |

These figures are identical to the Phase 4 reference run (same seed 14, same configuration
files, same calibration artifact), as expected since Phase 5 performs no new attack logic.

## SHA-256 verification

Computed by the CLI, independently re-verified by reopening each file and recomputing its
digest (also asserted in `tests/test_exporting_pipeline.py` and the real-dataset test):

| File | SHA-256 |
|---|---|
| `uscrn_attacked.parquet` | `e8cf097264b4736b6c837923a7d95fc6719158800836b2262a27e736b9a3b12c` |
| `detector_input.parquet` | `a6970ddb9ffe5e5247c57cd4d445df02c2e93a16d87be1662bf868e56207d089` |
| `ground_truth.parquet` | `5598d9500c89c91266dcf23cdfb7f900be09024a4b6b84bea209183a883ac978` |
| `attack_manifest.csv` | `4f41ac2ff1bb24dbc98af5174cd4486e3a84124670a2952cb7ac7d366d645446` |
| `data_contract.json` | `2a40dbdd5f3f5d467964e1800f91460c0cba3ae8f1c4e8f9a32a5d7bcb4f32a1` |
| `run_provenance.json` | `1d97cc9bb6e96bf068ed5e87ed5ebaf013b5759970e74dee933e31079890fe75` |
| `HANDOVER.md` | `370a382840a6bb38bb23ea127d0d73b55c8cc894f0da61e4950349622c9e3ded` |

All seven are also listed in the run's own `SHA256SUMS.txt`, which additionally covers itself
being absent from `run_provenance.json`'s `output_files` (no circular hashing). The source
Parquet's SHA-256 was identical before and after the export.

## Detector input leakage checks

Independently reopening `detector_input.parquet` confirmed its columns are exactly:
`station`, `time`, `partition`, `SOIL_MOISTURE_5` (the attacked, observable value),
`LATITUDE`, `LONGITUDE`, `T_HR_AVG`, `T_MAX`, `T_MIN`, `P_CALC`, `SOLARAD`, `SUR_TEMP`,
`RH_HR_AVG`, `SOIL_MOISTURE_10/20/50/100`, `SOIL_TEMP_5/10/20/50/100` — no `is_attacked`,
`attack_type`, `attack_id`, second clean-target column, or `flag_incoherence_verticale`.

## Manifest and ground-truth reconciliation

- `ground_truth.is_attacked.sum() == 9679 == manifest[status=="applied"].modified_count.sum()
  == summary.committed_total == ledger.total_committed`.
- `(~ground_truth.is_evaluable).sum() == 5401`, matching the missing-target count, and every
  one of those rows has `attacked_target` null and `is_attacked = False`.
- Every row in `uscrn_attacked.parquet`'s 23 original columns is exactly equal
  (`assert_series_equal`) to the corresponding column of the loaded source frame.
- Per-family committed counts in `ground_truth.attack_type.value_counts()` match
  `ExecutionSummary.per_family[*].committed` exactly (spike 936 / bias 2,798 / drift 2,839 /
  replay 2,416 / stealth 690).

## Complete output directory path

`attack-engine/datasets/generated/run-20261022T211937Z-seed14-a81da0/` (git-ignored by
`datasets/generated/*`, consistent with the project's existing policy of never committing
generated data).

## Remaining limitations

- Export is deterministic given a fixed dataset, configuration and dependency environment, but
  (as with Phases 1-4) NumPy's bit-stability across versions is not guaranteed; pin `uv.lock`
  for bit-identical re-exports.
- The manifest's status model includes `cancelled` for forward-compatibility, but this phase
  never produces it (every planned window today resolves to `applied`, `rejected`, or
  `skipped` for a non-executable family, which does not currently occur either).
- `predictions.parquet` (the AI team's expected output) is documented in `HANDOVER.md` but not
  implemented or benchmarked; per the task scope, Phase 6 benchmarking is out of scope here.
- The export reuses Phase 1's partition logic and Phase 3/4's execution results verbatim; it
  inherits all of their documented limitations (see `README.md` and `PHASE4_REPORT.md`).

## Readiness for AI handover

The bundle is self-contained and independently verifiable: `SHA256SUMS.txt` lets any recipient
confirm every file, `data_contract.json` documents the exact schema and allowlist machine-
readably, and `HANDOVER.md` explains purpose, audience, label semantics, the detector feature
allowlist, the ground-truth access policy (test labels restricted to the evaluator until
predictions are frozen), and how to join `predictions.parquet` back to `ground_truth.parquet`.
`detector_input.parquet` is ready to hand to the AI team as-is; `ground_truth.parquet` for
`test` should stay with the evaluator until predictions are frozen.
