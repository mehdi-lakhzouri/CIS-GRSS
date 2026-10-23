# Coordinated Multi-Variable Stealth — Report

Coordinated stealth attacks that modify several USCRN variables at once, built on the
Multi-Target Foundation and reusing the existing executor, scheduler and budget ledger. The
single-variable families and the seed-14 baseline are unchanged.

## Scenarios

A new attack family, `multi_stealth`, modifies the soil-moisture target plus one or more
coupled context variables within one shared window. Three campaign configs under
`configs/campaigns/` differ only in `secondary_columns`:

| Scenario | File | Secondary variables |
|---|---|---|
| Multi-Depth Stealth | `campaigns/multi_depth.yaml` | `SOIL_MOISTURE_10` |
| Moisture + Rainfall | `campaigns/moisture_rain.yaml` | `P_CALC` |
| Coordinated Stealth | `campaigns/coordinated.yaml` | `SOIL_MOISTURE_10`, `P_CALC` |

All three share `campaigns/scheduling.yaml` (family `multi_stealth`, budget scope
`validation`/`test`). Rainfall modification is an explicit, opt-in attacker capability
(`P_CALC` in `secondary_columns`), never a universally compromised source.

## How it works

- The soil-moisture (`SOIL_MOISTURE_5`) channel **reuses the existing single-variable stealth
  transformation and its frozen train calibration unchanged** (constrained local FDI against a
  train-only station ridge; no detector consulted).
- Secondary variables are driven by lightweight couplings fitted **only on the clean train
  partition (2022–2023)**:
  - `SOIL_MOISTURE_10` follows the 5 cm injected offset through a per-station depth slope
    (`cov(sm5,sm10)/var(sm5)`, clamped to `[0, depth_coupling_max_ratio]`), lagged by
    `depth_coupling_lag_hours` and continuity-limited by `depth_max_step_change`. Deeper
    moisture responds, damped and delayed.
  - `P_CALC` receives a bounded rain pulse placed `rainfall_lag_hours` **before** a moisture
    rise, capped by the per-station train rainfall ceiling and `rainfall_max_mm`.
- **Physical bounds** are enforced per variable (soil moisture `[0, 0.70]`, rainfall
  `[0, rainfall_max_mm]`); **missing values are preserved** — a missing 10 cm value stays
  genuine and rain is never fabricated into a missing slot. No constraint is relaxed to
  consume a quota; a rejected soil-moisture channel rejects the whole coordinated event.
- **Budget**: the ledger still counts one soil-moisture cell per attacked row, so the
  configured budget is never exceeded. Secondary cells are tracked in `committed_cells`,
  separate from `committed_rows`/`committed_total`, with exact per-variable `changed_mask`s.
- **Scheduling**: the campaign budget scope is `validation`/`test`, so no window is planned in
  the train calibration partition and the baseline's train-partition stealth rejections are
  avoided entirely.
- Window lengths, attack frequency (budget), magnitudes, couplings and the global seed are all
  configurable; execution is deterministic for a fixed dataset, config and dependency set.

## Real USCRN results (seed 14, scope validation+test, budget 0.03)

Planned 82, applied 71, rejected 11 (identical window selection across scenarios, since the
soil-moisture channel is unchanged); quota deficit 570; mean primary impact 0.00971 m³/m³.

| Scenario | Applied | Rows | Cells | Per-variable cells |
|---|---|---|---|---|
| Multi-Depth | 71 | 2,538 | 5,005 | sm5 2,538; sm10 2,467 |
| Moisture + Rain | 71 | 2,538 | 2,574 | sm5 2,538; P_CALC 36 |
| Coordinated | 71 | 2,538 | 5,041 | sm5 2,538; sm10 2,467; P_CALC 36 |

Rejections (per scenario): `empty_contextual_feasible_range` 8, `missing_required_context` 2,
`no_feasible_quantized_value` 1. Rainfall cells are deliberately sparse (36): a pulse is placed
only when a moisture rise coincides with a genuine, finite rainfall slot at the configured lag
and the per-station ceiling allows headroom — plausibility is preserved rather than forcing
unrealistic rain. `committed_cells > committed_rows` confirms coordinated multi-variable edits;
`committed_total == committed_rows == 2,538` confirms the soil-moisture budget is respected.

## Backward compatibility

- **Seed-14 baseline unchanged: 9,679 modifications including 690 stealth** (verified by the
  real-dataset regression). Existing families, configs, calibration artifact and Phase 5
  exports are untouched; multi-variable campaigns use separate config files.
- **Original dataset unchanged**: input SHA-256 `46b98115…b98421` identical before and after the
  full suite (the executor verifies this with a frame checksum).

## Tests

- `tests/test_multistealth.py` (12): depth candidate (lagged, damped, bounded, missing
  preserved, zero-slope no-op); rainfall pulse (precedes the rise, respects the ceiling, never
  fills a missing slot); train-only calibration with an explicit no-leakage check (corrupting
  validation/test rows does not change the fitted slope); coordinated multi-column event
  (multiple columns, `modified_count` counts only the primary, cells ≥ count, unique cell
  keys); determinism; rejected-primary propagation; executor integration (cells vs rows vs
  budget, no conflicting writes, determinism, validation/test-only placement, frame unchanged).
- `tests/test_real_dataset.py`: a coordinated real-dataset run (read-only, deterministic,
  cells > rows) plus the unchanged seed-14 baseline.
- Full suite: **416 passed** (was 403), ruff + format clean, real-dataset hash unchanged.

## Limitations (intentional)

- The soil-moisture channel is the gate: if it is rejected (empty feasible range, missing
  context, infeasible quantization), the whole coordinated event is rejected. Secondary
  variables never rescue a rejected primary.
- Depth coupling is a per-station linear slope; rainfall uses a single bounded lagged pulse per
  rise. These are lightweight, plausible couplings, not calibrated hydrological models.
- Secondary calibration is fitted on-the-fly from the train partition each run (deterministic);
  it is not yet frozen to a separate JSON artifact.
- Per the task scope, **no export, AI benchmarking, ERA5 integration, Sensor Emulator or
  cryptography** was implemented. The variable-level ground truth remains the foundation's
  builder and is not wired into the default export pipeline; coordinated secondary edits are
  produced and validated in memory only.

## Next

Freeze the secondary calibration as a versioned artifact and wire the variable-level ground
truth into the export pipeline when export changes are in scope.
