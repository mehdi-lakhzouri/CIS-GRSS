# Multi-variable FDI across all five families

Validated on 23 October 2026. Spike, Bias, Drift, Replay and Stealth now support
`SOIL_MOISTURE_5`, `SOIL_MOISTURE_10` and `P_CALC`, individually or in any combination.
The existing scheduler, executor, ledger, result models and train calibration are reused.
The attack-family enum and registry contain exactly five families. `multistealth.py` is a
secondary-variable strategy called by Stealth; it is not a sixth family.

The original seed-14 campaign still produces **9,679 modified cells/rows, including 690
Stealth**, with the original per-family totals: Spike 936, Bias 2,798, Drift 2,839,
Replay 2,416, Stealth 690. The baseline YAML files and frozen calibration are unchanged.

## Architecture and behavior

Every family's parameters accept `target_columns`. The first selected variable supplies the
backward-compatible primary result arrays; `extra_columns` contains the other modified
channels. The executor reads that actual primary column, rather than always reading 5 cm
moisture. The three single-variable choices, three pairs, triple and reordered triple are
tested for every transformation.

| Family | Coordinated behavior |
|---|---|
| Spike | Draw positions and sign once per event; apply synchronized spikes with each variable's explicitly configured magnitude. |
| Bias | Apply independently configured constant offsets over the same window. Reject bounds violations and offsets that become nonconstant or partially disappear after quantization. |
| Drift | Use one ramp in elapsed hours, with independent final magnitudes and slope limits. Coordinated ramps also check their realized slopes after quantization. |
| Replay | Choose one strictly earlier, time-aligned block in the same station and partition, finite and within bounds across every selected channel. Copy genuine source values exactly for coordinated and non-default single-variable requests. Preserve destination missing values. |
| Stealth | Reuse the frozen train-only 5 cm contextual rollout, train-fitted damped/lagged depth response and lagged bounded rainfall pulses. For a secondary-only selection, use the 5 cm rollout as a latent reference and commit only selected channels. |

For Spike, Bias and Drift, `magnitude` / `final_magnitude` applies in the first selected
variable's units; every additional variable requires its own `secondary_magnitudes` entry.
No rainfall magnitude is inferred by copying a moisture magnitude. Drift additionally accepts
`secondary_max_slopes_per_hour`. Replay requires no additive magnitude.

Stealth accepts the same `target_columns` interface. The earlier `secondary_columns` spelling
remains compatible with the 5 cm primary strategy; conflicting selections are configuration
errors. A missing reference, calibration, required context or useful secondary response leads
to an explicit rejection.

## Atomicity, masks and physical constraints

- **Atomic event:** every selected variable must change at least one finite cell. If any
  selected channel is infeasible or has zero effect, reject the entire event and cancel its
  reservation. Missing secondary observations remain unchanged and unlabelled. A window with
  a missing primary observation is rejected, preserving the original primary policy.
- **Exact cell truth:** masks are computed from finite clean/final values and exact inequality.
  `changed_variables` and `per_variable_modifications` identify actual changes; rejected events
  report zero changes. `modified_count` retains its primary-channel meaning; use
  `modified_cells` and `modified_rows` for coordinated accounting.
- **Bounds and precision:** soil moisture defaults to `[0, 0.70]` m3/m3 and rainfall to
  `[0, 200]` mm per hourly observation. `variable_constraints` can tighten bounds and configure
  precision separately. These are project experiment limits, not universal hydrological laws.
  Coordinated Spike/Drift requests that exceed bounds are rejected rather than silently clipped.
- **Continuity:** configurable `max_step_per_hour` and `max_offset_rate_per_hour` check finalized
  values inside the window and at contiguous entry/exit boundaries, including earlier emitted
  attacks. Missing observations, time gaps, station boundaries and partition boundaries break
  continuity. Secondary lags use elapsed hours rather than array positions.
- **Cross-variable constraints:** `relationships` enforce
  `lower <= left - scale * right <= upper` on values or injected offsets. The scale converts
  between the operands' units when necessary. Checks cover affected rows with both operands
  finite; they never fabricate a missing measurement. The supplied campaigns constrain the
  5 cm minus 10 cm moisture difference to `[-0.4, 0.4]` m3/m3.
- **Compatibility:** original single-target saturation and precision behavior is retained for
  the reference campaign, including the legacy Drift ramp. New per-variable constraints can
  explicitly tighten that behavior. This preserves the audited seed-14 baseline.

## Scheduling and budgets

`family_parameters.<family>.variable_count` must equal the selected target count. The executor
checks both family totals and each event's reservation before executing. `eligibility_target`
selects the finite-observation denominator and eligible segments; its default remains
`SOIL_MOISTURE_5`. Set it to `P_CALC` for a rainfall-only campaign that should not depend on
finite 5 cm moisture.

`budget_unit: cells` reserves `window_length * variable_count` and commits the sum of all
per-variable masks. `budget_unit: rows` reserves the window length and commits the union of
modified rows. In both cases, cell counts and per-variable counts are always reported.
The fractional quota remains `round_half_up(attack_budget * finite_denominator)` in the
declared unit; it is not multiplied by the number of channels.

Independent `max_modified_rows` and `max_modified_cells` caps apply under either policy. An
event that would exceed either cap is rejected atomically. Unused capacity is released and
quota deficits are reported rather than filled by relaxing physical checks. The scheduler
retains its conservative non-overlapping windows; the executor also rejects duplicate writes
to a `(row, variable)` cell before changing write-tracking state.

Family summaries now report reservations and commits in the same configured budget unit,
plus separate modified rows, modified cells and per-variable totals. CLI generation JSON
includes the policy, caps, counts and deficits.

## Separate campaigns and real USCRN execution

Each directory contains independent `attacks.yaml` and `scheduling.yaml` files:

- [Spike](configs/campaigns/spike/attacks.yaml), [scheduling](configs/campaigns/spike/scheduling.yaml)
- [Bias](configs/campaigns/bias/attacks.yaml), [scheduling](configs/campaigns/bias/scheduling.yaml)
- [Drift](configs/campaigns/drift/attacks.yaml), [scheduling](configs/campaigns/drift/scheduling.yaml)
- [Replay](configs/campaigns/replay/attacks.yaml), [scheduling](configs/campaigns/replay/scheduling.yaml)
- [Stealth](configs/campaigns/stealth/attacks.yaml), [scheduling](configs/campaigns/stealth/scheduling.yaml)

All select the three variables, seed 14, validation/test placement, a 0.03 cell budget,
`variable_count: 3`, a row cap of 1,100 and a cell cap of 3,108. The finite denominator is
103,611, giving **3,108 cell quota units per independent campaign**. The train partition
remains genuine. Runs were performed twice per family and reproduced identical event/value/
mask SHA-256 fingerprints, statuses and per-variable counts.

| Family | Planned | Applied | Rejected | Reserved cells | Modified rows | Modified cells | Quota deficit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spike | 308 | 149 | 159 | 3,090 | 266 | 798 | 2,310 |
| Bias | 37 | 19 | 18 | 3,030 | 634 | 1,902 | 1,206 |
| Drift | 24 | 8 | 16 | 2,883 | 264 | 792 | 2,316 |
| Replay | 46 | 15 | 31 | 3,006 | 475 | 1,057 | 2,051 |
| Stealth | 33 | 16 | 17 | 2,961 | 477 | 969 | 2,139 |

| Family | SOIL_MOISTURE_5 cells | SOIL_MOISTURE_10 cells | P_CALC cells |
|---|---:|---:|---:|
| Spike | 266 | 266 | 266 |
| Bias | 634 | 634 | 634 |
| Drift | 264 | 264 | 264 |
| Replay | 472 | 468 | 117 |
| Stealth | 477 | 461 | 31 |

Quota deficit includes both unreserved capacity and reserved capacity that did not become a
modification. Unreserved cells are respectively 18, 78, 225, 102 and 147. Replay and Stealth
modify different subsets of rows in different channels, so modified rows are the union rather
than the primary count or a cell count divided by three.

Rejections are explicit:

| Family | Reasons |
|---|---|
| Spike | 158 bounds violations; 1 zero-effect selected channel |
| Bias | 17 bounds violations; 1 zero-effect selected channel |
| Drift | 16 bounds violations |
| Replay | 29 zero-effect selected channels; 2 temporal constraint violations |
| Stealth | 15 zero-effect selected channels; 2 empty contextual feasible ranges |

Rainfall is often zero in a genuine replay source, or has no feasible contextual pulse. The
atomic policy rejects those events instead of claiming that all three channels changed.
Negative coordinated offsets at dry rainfall observations also produce bounds rejections.
These deficits are expected consequences of the selected experiment parameters.

Machine-readable statistics, every event's changed-variable counts and rejection details,
configuration hashes and reproducibility fingerprints are in
[multivariable_all_families_results.json](validation/multivariable_all_families_results.json).

## Validation and integrity

**507 tests passed**, with `ATTACK_ENGINE_REAL_DATASET` pointing at the reference delivery;
no real-data tests were skipped. Ruff lint and formatting checks are clean.

The new synthetic suite covers every family/target selection and reordering, synchronized
spikes, constant bias, elapsed-time ramps and lags, per-variable slopes and quantization,
missing-value preservation, infeasible sources, source alignment, exact replay values,
cross-variable rejection, entry/exit continuity, previous event offsets, all-family executor
integration, mixed target counts, both budget units, both hard caps and atomic conflict checks.

Real-USCRN tests execute every multi-variable family twice and verify exact masks, bounds,
source values, per-variable counts, unique cell writes, placement, reproducibility and input
immutability. Existing tests verify frozen primary calibration against the clean train data
and poison all future target/context measurements to detect leakage. Secondary calibration
also remains identical after poisoning validation/test channels. The standalone audit
independently reconstructs each final in-memory frame and checks global temporal edges,
relationships, train preservation and all ledger commits.

SHA-256 checks protect **55 original files**, including the USCRN delivery, archive, baseline
configuration, calibration and existing generated-directory placeholder. No prior export
bundle was present in that directory. The full protected-file inventory and initial hashes are
in [multivariable_integrity_before.json](validation/multivariable_integrity_before.json).

| Protected artifact | Unchanged SHA-256 |
|---|---|
| `USCRN_livraison/data/uscrn_clean.parquet` | `46b98115424bb03af28d31bcb96b68fedd537a609bf97d4199a653a376b98421` |
| `USCRN_livraison/data/uscrn_clean.csv` | `06be73a338b79f50ded3a86eb3890699ed08d260ade86e64c7c4f7ebbb84e3d4` |
| `calibrations/uscrn_train.json` | `2b758682269f7a102be34f99070b91e09b86417b38ab3096a823ce2c9dedea78` |

Reproduce from `attack-engine` in PowerShell:

```powershell
$env:ATTACK_ENGINE_REAL_DATASET = (Resolve-Path ../USCRN_livraison/data/uscrn_clean.parquet).Path
uv run pytest -q --basetemp=.pytest-tmp-multivariable-check --tb=short
uv run ruff check .
uv run ruff format --check .
uv run python scripts/validate_multivariable.py
```

Generate a single campaign in memory:

```powershell
uv run attack-engine generate --input ../USCRN_livraison/data/uscrn_clean.parquet --config configs/dataset.yaml --attack-config configs/campaigns/spike/scheduling.yaml --attack-params configs/campaigns/spike/attacks.yaml --json
```

The standalone validator writes statistics/integrity JSON only. Generation returns candidates
and masks in memory. Existing export regression tests use isolated temporary directories.

## Scope boundary and practical limits

Generation and validation are complete. No multi-variable dataset export, ERA5 integration,
AI evaluation, Sensor Emulator or cryptographic feature was implemented. The existing export
pipeline and its single-target regression remain unchanged by this task.

Secondary Stealth uses the existing lightweight depth/rainfall couplings, not a full water
balance model, and may reject a large share of requested events. Its contextual reference uses
the original clean contemporaneous measurements; this work does not establish detector
evasion against the resulting modified context. Physical feasibility is defined by the
explicit experiment constraints. Temporal limits are configurable to preserve compatibility,
and enabled in all five supplied campaigns. The scheduler does not retry rejected execution
events to fill deficits. Secondary calibration remains train-only and fitted in memory; the
original frozen primary artifact is reused unchanged.
