# Multi-Target Foundation — Report

Foundation for multi-variable False Data Injection in `attack-engine`. This change makes the
typed models, budget accounting, overlap detection and export contracts **ready** for attacks
that modify several variables at once, **without implementing any multi-variable generation**.
Every current transformation still modifies the single target `SOIL_MOISTURE_5`.

## Scope and guarantees

- **No new attack behaviour.** Spike, bias, drift, replay and stealth are untouched. The
  seed-14 reference run still produces **9,679 modifications, including 690 stealth**
  (spike 936, bias 2,798, drift 2,839, replay 2,416, stealth 690).
- **Original dataset untouched.** Input SHA-256 `46b98115…b98421` is identical before and
  after the full suite (now checked by a per-frame checksum in the executor, not just asserted).
- **Backward-compatible.** All existing configs, calibration, Phase 5 exports and their on-disk
  formats are unchanged. New model fields are additive with single-target defaults.

## Changes

### 1. Multi-target typed models (`attacks/models.py`)
- `DEFAULT_TARGET_COLUMN` and `CellKey = tuple[int, str]` (a cell is one `(row, variable)`;
  since a row label uniquely identifies a `(station, time)` observation, a cell key is exactly
  a `(station, time, variable)` coordinate).
- `ColumnModification`: one variable's `clean/candidate/changed_mask` plus `modified_count`,
  `modified_labels`, `cell_keys()`.
- `AttackRequest`: adds `target_columns` (empty ⇒ the default target) with
  `resolved_target_columns` / `primary_target_column`, plus an `extra_clean` map reserved for
  future per-variable clean series.
- `AttackResult`: adds `target_columns` and `extra_columns`, and a uniform `columns` view
  (primary target + extras). New accounting: `modified_cells`, `modified_rows`,
  `modified_row_labels()`, `modified_cell_keys()`. For a single-target result,
  `modified_cells == modified_rows == modified_count`.
- Result builders (`attacks/base.py`) propagate `target_columns` from the request.

### 2. Scheduling & budget (`attacks/executor.py`)
- The scheduler and `BudgetLedger` are reused unchanged.
- The overlap guard now keys on **cells** `(row, variable)` instead of rows, so two future
  multi-variable events may touch different variables of the same row, while a genuine
  double-write of one variable still raises `ExecutionError`.
- `ExecutionSummary` distinguishes `committed_total` (budget cells), `committed_cells` and
  `committed_rows`. All three equal 9,679 for the current single-target run; they diverge only
  once one event modifies several variables of a row. Surfaced in the `generate` CLI output.

### 3. Ground-truth & export contracts (`exporting/`)
- `DatasetContract` gains `target_columns`, `attacked_columns` (clean→attacked name map) and
  `variable_ground_truth_columns`, defaulted to the single-target values.
- New `GroundTruthCellRecord` model and `VARIABLE_GROUND_TRUTH_COLUMNS` define the
  **variable-level** label schema (`station, time, partition, column, clean_value,
  attacked_value, is_attacked, attack_type, attack_id, is_evaluable`), alongside the existing
  **row-level** `GroundTruthRecord`.
- `build_cell_ground_truth(ground_truth, target_column)` projects the row-level ground truth to
  the per-variable (cell) table. It is **not** written by the default pipeline, so existing
  exports are byte-for-byte unchanged; it is available and tested for future use.
- `attack_type` and `attack_id` remain tracked per cell; `attacked_columns` is tracked in the
  contract.

### 4. Tests
- `tests/test_multitarget_foundation.py` (11 tests): request/result default target, cells vs
  rows (including a hand-built two-variable result proving `modified_cells` ≠ `modified_rows`
  semantics), cell-overlap guard (same row + variable collides; same row + different variables
  do not), summary rows/cells equality for the single-target run, contract fields, and the
  variable-level ground-truth projection validated against `GroundTruthCellRecord`.
- `tests/test_real_dataset.py`: the seed-14 regression now also asserts
  `committed_cells == committed_rows == 9,679`, `stealth.committed > 0`, and that the derived
  cell ground truth has 9,679 rows all on `SOIL_MOISTURE_5`.

## Results

- `uv run ruff check .` — clean. `uv run ruff format --check .` — clean.
- `uv run pytest` — **403 passed** (was 387; +16), 5 real-dataset tests run under
  `ATTACK_ENGINE_REAL_DATASET`.
- Real USCRN regression (seed 14): committed 9,679 (stealth 690); `committed_cells ==
  committed_rows == 9,679`; input SHA-256 unchanged.

## Limitations (intentional)

- No transformation produces multi-variable results yet; `extra_columns` is always empty and
  `extra_clean` is unused. Coordinated multi-variable stealth is the next task.
- The default export pipeline still writes the single-target row-level artifacts; the
  variable-level table is a foundation builder, not yet wired into `export_run`.
- Budget quotas are still allocated per family in cells; a future multi-variable policy may need
  to decide whether a multi-cell event counts against one or several variables' budgets.

## Next

Implement **Coordinated Multi-Variable Stealth** on top of this foundation (per-variable
candidate generation, cross-variable feasibility, and wiring the variable-level ground truth
into the export pipeline). Not started.
