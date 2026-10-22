# attack-engine

Attack Engine for **TSYP14 — Ground Truth Under Attack — Track 2: False Data Injection**.

The engine will generate controlled, labelled adversarial modifications of environmental
sensor data (NOAA USCRN hourly soil moisture) to evaluate anomaly detection systems.

> **Status: Phases 1–5 — dataset foundation, planning, five attack families and dataset export.**
> The project loads, validates, profiles and hashes the dataset (Phase 1), plans attack
> windows and budget allocation (Phase 2), generates candidate attack values in memory
> for spike, bias, drift, replay and context-aware stealth (Phases 3–4), and exports a
> versioned ground-truth, provenance and dataset bundle downstream of that execution
> (Phase 5). It does **not** detect anomalies, sign data, or expose any API or frontend.
> The `generate` command computes candidate values in memory and reports what changed; it
> never writes a file and never modifies the source observations. The explicit `calibrate`
> command writes only a small train-only JSON reference artifact, and `export` writes a
> new, uniquely identified run directory under `datasets/generated/`.

## Technology stack

Python 3.12+, uv, Pandas, NumPy, PyArrow (Parquet), Pydantic (configuration and structured
results), PyYAML (configuration files), Pytest, Ruff.

## Project structure

```text
attack-engine/
├── configs/
│   ├── dataset.yaml              # reference-delivery profile, expected hash, partitions
│   ├── scheduling.yaml           # seed, budget, scope, families, weights, window bounds
│   └── attacks.yaml              # per-family magnitudes, sign policy, quantization
├── calibrations/uscrn_train.json  # explicit train-only reference artifact (Phase 4)
├── datasets/{raw,processed,generated}/   # local data folders (contents git-ignored)
├── src/attack_engine/
│   ├── cli.py                    # argument parsing and output formatting only
│   ├── config.py                 # dataset YAML -> validated, immutable AppConfig
│   ├── dataset/
│   │   ├── schema.py             # columns, types, units, ranges, keys, timestamp semantics,
│   │   │                         # validation profiles, partition definitions
│   │   ├── loader.py             # read-only Parquet loading + provenance metadata
│   │   ├── temporal.py           # interval-end semantics, gaps, partition assignment
│   │   ├── validator.py          # structural, numerical, temporal and partition checks
│   │   └── profiler.py           # descriptive JSON-compatible statistics
│   ├── scheduling/               # Phase 2: planning only, no attack values
│   │   ├── models.py             # AttackWindow, ScopeCell, families, rejection reasons
│   │   ├── config.py             # scheduling YAML -> validated SchedulingConfig
│   │   ├── identifiers.py        # SHA-256 + SeedSequence deterministic generators
│   │   ├── eligibility.py        # finite-target denominators and continuous segments
│   │   ├── budget.py             # rounding, largest-remainder allocation, reservation ledger
│   │   └── scheduler.py          # AttackWindowScheduler -> ScheduleResult
│   ├── attacks/                  # Phases 3-4: transformations and in-memory execution
│   │   ├── models.py             # AttackRequest/Result/Diagnostics, bounds, settings
│   │   ├── config.py             # attack YAML -> validated per-family parameters
│   │   ├── base.py               # quantize, clip, changed-mask, result builders
│   │   ├── spike.py bias.py drift.py replay.py   # the four transformations
│   │   ├── stealth.py             # sequential contextual projection, Phase 4
│   │   ├── calibration.py         # station ridge model, causal features, JSON artifact
│   │   ├── context.py             # read-only context from eligibility + earlier emissions
│   │   ├── registry.py           # family -> transformation dispatch
│   │   └── executor.py           # AttackExecutor -> ExecutionSummary (commits to ledger)
│   ├── exporting/                # Phase 5: ground truth, provenance, dataset export
│   │   ├── models.py             # ExportConfiguration/Run/Result, DatasetContract, provenance
│   │   ├── contracts.py          # explicit detector-input allowlist and contract builder
│   │   ├── builder.py            # attacked target, audit/ground-truth/detector-input frames
│   │   ├── manifest.py           # event-level attack_manifest.csv rows
│   │   ├── validation.py         # the Phase 5 scientific/consistency checks
│   │   ├── provenance.py         # run_provenance.json (never hashes itself)
│   │   ├── handover.py           # HANDOVER.md generation
│   │   ├── writer.py             # atomic temp-dir-then-rename writing, SHA256SUMS.txt
│   │   └── pipeline.py           # export_run(): the single orchestration entry point
│   └── utils/hashing.py          # streaming SHA-256 and integrity comparison
└── tests/                        # deterministic synthetic fixtures (+ opt-in real-data tests)
```

## Setup

```bash
# Install uv: https://docs.astral.sh/uv/
cd attack-engine
uv sync            # creates .venv and installs locked runtime + dev dependencies
```

## Dataset placement

The engine reads any path given with `--input`; it never writes to, cleans or reorders the
source. Two options:

- point `--input` at the delivered file, e.g. `../USCRN_livraison/data/uscrn_clean.parquet`;
- or copy it into `datasets/raw/` (git-ignored) and use `datasets/raw/uscrn_clean.parquet`.

Only Parquet is supported.

## Validate

```bash
uv run attack-engine validate --input ../USCRN_livraison/data/uscrn_clean.parquet
uv run attack-engine validate --input ../USCRN_livraison/data/uscrn_clean.parquet --json
```

Options:

| Option | Meaning |
|---|---|
| `--config PATH` | YAML configuration; defaults to `configs/dataset.yaml` when it exists in the working directory |
| `--validation-profile real\|fixture` | `real`: reference-delivery expectations (row/column counts, six stations, complete years), expected SHA-256 and partitions from the config. `fixture`: generic schema rules only. Default: `real` if a config is found |
| `--expected-sha256 HEX` | Expected digest, overriding the config |
| `--json` | Machine-readable output on stdout |

Exit codes: `0` passed, `1` validation errors or SHA-256 mismatch, `2` usage, input or
configuration error (missing file, unsupported format, unreadable Parquet, invalid config).

## Profile

```bash
uv run attack-engine profile --input ../USCRN_livraison/data/uscrn_clean.parquet [--json]
```

Reports rows, columns, stations, observation period, rows and missing values per station
and column, target min/max/mean/sample std and coverage, duplicate keys, temporal gaps,
year-boundary rows, partition row counts, and validation errors/warnings. No individual
observations are printed. Statistics are descriptive and are not evidence of any attack.

## Plan (Phase 2)

Plans *where* attacks could be placed and how the budget is divided. It reserves windows and
allocates a budget; it does **not** compute, apply or write any modified value.

```bash
uv run attack-engine plan \
  --input ../USCRN_livraison/data/uscrn_clean.parquet \
  --config configs/dataset.yaml \
  --attack-config configs/scheduling.yaml [--json]
```

`--config` supplies the partition definitions and the validation profile; planning first
runs validation and refuses to proceed on an invalid dataset. `--attack-config` supplies the
seed, budget, scope, families and window bounds. The command prints: the seed; selected
stations and partitions; the requested budget and absolute quota; per-family quotas,
allocated and reserved counts, and window counts; a sample of planned windows; and the
unfulfilled allocations with traceable rejection reasons. `--json` emits the full plan,
including every window and rejection.

Exit codes match the other commands: `0` success, `1` the dataset fails validation (or a
strict `insufficient_policy=fail` deficit), `2` input/configuration error (missing file,
invalid scheduling config, or a `budget_scope` partition not in the plan).

### Budget accounting

Five quantities are tracked separately and never conflated:

| Quantity | Meaning |
|---|---|
| requested | `round_half_up(attack_budget × N_finite)` over the scope; a `0` budget is strictly `0` |
| allocated | integer quota per (family, station, partition) cell, by largest-remainder split |
| reserved | capacity held by scheduled windows — **not** a count of attacked observations |
| committed | observations actually modified, committed by `generate`; reserved until then |
| unfulfilled | allocation that could not be reserved, with a reason per cell |

`N_finite` is the count of finite target observations in the scope, computed per cell before
any family constraint; continuity constraints reduce *eligible* positions but never shrink
the denominator. The ledger refuses to let reservations exceed a cell's allocation, so the
configured budget cannot be silently overshot. Cancelling or committing a reservation frees
the capacity it is not using.

## Generate (Phases 3–4)

Executes attacks over a plan **in memory** and prints a summary. It computes candidate
modified values for five executable families (spike, bias, drift, replay, stealth), commits the
number of observations actually modified to the budget ledger, releases unused or failed
reservations, and reports the deficit. It writes nothing and never changes the source data.

```bash
# Explicit calibration: fits only train and writes this one JSON artifact.
uv run attack-engine calibrate \
  --input ../USCRN_livraison/data/uscrn_clean.parquet \
  --config configs/dataset.yaml \
  --attack-params configs/attacks.yaml \
  --output calibrations/uscrn_train.json

# Generation reads the artifact, computes in memory and writes no dataset.
uv run attack-engine generate \
  --input ../USCRN_livraison/data/uscrn_clean.parquet \
  --config configs/dataset.yaml \
  --attack-config configs/scheduling.yaml \
  --attack-params configs/attacks.yaml [--json]
```

`--attack-params` (see `configs/attacks.yaml`) sets per-family magnitudes, sign policy and the
optional quantization grid. The transformations:

| Family | Behaviour |
|---|---|
| spike | shifts 1–3 positions (chosen without replacement) by `±magnitude`; clips to bounds |
| bias | adds a constant `±magnitude` over the whole window; rejects the event if the offset would leave `[0, 0.70]`, so the realized offset stays constant |
| drift | linear ramp in elapsed time reaching `±final_magnitude` at the last point; first point held; rejects windows too short or slopes above the limit |
| replay | copies an equal-length, hourly-contiguous, fully-finite earlier run of the same cell from the clean data; timestamps and context columns untouched |
| stealth | projects a proposed offset into a train-calibrated contextual band and physical/temporal constraints, feeding earlier finalized emissions into its history |

A position counts as **modified** only when the final value (after quantization and bounds
clipping) is finite and differs from the clean value, so a spike that reserves a window of six
but changes two positions commits two. The clean DataFrame is never mutated; the summary is an
in-memory structure only — attacked datasets, labels and manifests are deferred to a later phase.

### Numerical precision policy

An exactly zero requested perturbation keeps the original float **exactly**, including
values off the configured grid. When a quantum is enabled, clean and candidate values in
the same round-half-to-even bin also retain the exact clean value. A bin crossing remains
effective even if the requested delta is smaller than one quantum. Quantization is never
used to normalize untouched observations.

The changed mask is computed by exact comparison with the final, clipped values. There is
no fixed `1e-9` detection threshold: with `quantum: null`, even a one-ULP representable
difference counts. This prevents `0.141` on a `0.001` grid from becoming a false modification,
without suppressing real small attacks under a finer grid or with quantization disabled.
Bias feasibility allows only scale-dependent float64 arithmetic roundoff and checks the
offset across **all** window positions, including unchanged ones. Numerical parameters must
be finite.

### Capacity and unsupported families

The per-family summary separates `allocated`, `unreserved`, `reserved`, `max_modifiable`,
`committed`, `deficit` and `unfulfilled_quota`. The legacy `deficit` field means unused
reservations (`reserved - committed`); `unfulfilled_quota` means the complete shortfall
(`allocated - committed`). `max_modifiable` is a structural upper bound before physical
rejection, clipping and quantization, not a promise that it will be achieved.

All five families execute when configured. A missing Stealth parameter block or calibration
artifact is an explicit error, irrespective of the unsupported-family policy. Future
unsupported families retain an explicit policy:

- `--on-unsupported-family skip` (default, compatible with existing plans): release the
  unsupported reservations and report their families, window counts and quota shortfall in
  text and JSON. No silent skipping.
- `--on-unsupported-family error`: fail with exit code 1 before any reservation is committed
  or released. The Python API exposes the same `UnsupportedFamilyPolicy` enum.

Generation also refuses a configured source SHA-256 mismatch before execution. The Phase 2
planner and its scheduling configuration are preserved. It can still reserve Stealth windows
in train; Phase 4 rejects those windows explicitly and reports the unfulfilled quota.

The current real-data Spike plan reserves 1,845 positions in 534 windows, but can modify at
most 1,324 of them with `max_points: 3`. The budget still means modified observations. A
minimal compatible option is to set only Spike's `min_window` and `max_window` to `1`:
each event then targets one observation. A separate in-memory experiment achieved the
1,845 Spike quota with that override; the checked-in configuration was not changed.

See [PHASE3_AUDIT.md](PHASE3_AUDIT.md) for the reconciled real-data totals, the audit of every
applied Drift window, the numerical regressions and validation commands.

## Context calibration and Stealth (Phase 4)

The reference model is **not the final AI anomaly detector**. No detector prediction or
score is used. A standardized ridge regression is fitted independently for each station
using clean observations assigned to the explicitly named `train` partition only.
Validation and test rows are sliced out **before** building features. Normality here means
the supplied clean experimental reference; it is not a claim that field data are error-free.

The default feature vector contains `SOIL_MOISTURE_10`, hourly accumulated rain `P_CALC`
(mm), `SOIL_TEMP_5`, `T_HR_AVG`, the previous observed target, its trailing 24-hour mean,
and sine/cosine seasonal and UTC end-of-hour features. The history excludes the current
target. Missing targets or time gaps reset the history. Normalization, coefficients and the
99th percentile of absolute in-sample training residuals (with an explicit floor) are fitted
on train only. The station model has no implicit pooled fallback.

`calibrations/uscrn_train.json` records schema version, train boundaries, target, feature
order, history/interval lengths, hyperparameters, station coefficients and normalization,
residual tolerances, usable row counts, fitting time ranges and a fingerprint of the train
inputs. Its `calibration_sha256` hashes sorted compact JSON **excluding that digest field**;
it is a content identity, distinct from the hash of the indented JSON file. Loading verifies
the digest and compatibility with feature, partition, history and calibration settings.
Saving refuses to replace different content; an identical re-calibration is idempotent.
`calibration_path` is resolved relative to its attack-parameters YAML. No fitting happens
implicitly during `generate`.

Default Stealth execution is restricted to validation/test strictly after calibration.
Train windows are rejected as `stealth_partition_not_authorized`; this phase does not
support a chronological train-prefix injection mode. Their quota is not silently moved.
For each attempt, time-step `t` uses current untouched context and **previous emissions**:
its own finalized candidates inside a window, and earlier committed values from any family
before that window. The executor processes windows chronologically and returns results in
the original plan order. The clean DataFrame is never used as a hidden target-history oracle.

The admissible interval intersects the physical bounds, `mu_t ± residual_multiplier ×
station_tolerance`, the permitted emitted-value step and the permitted offset movement.
The drawn magnitude is a desired offset; projection may reduce or adjust it within the
explicit constraints. Feasible grid values are selected before using the common finalizer;
all constraints and the minimum impact are checked on finalized values. The mask definition
and numerical regressions from Phase 3 remain unchanged.

| Parameter | Meaning |
|---|---|
| `magnitude_min`, `magnitude_max`, `sign` | Desired signed offset, sampled from the deterministic per-window RNG |
| `residual_tolerance_multiplier` | Width of the frozen station reference band |
| `max_change_rate` | Maximum injected-offset change per **actual elapsed hour** |
| `max_step_change` | Maximum emitted-measurement change per elapsed hour on context-eligible observations |
| `max_cumulative_drift` | Sum of absolute changes in the injected offset across the event, starting from zero; also bounds its absolute size |
| `min_attack_impact` | Minimum mean absolute deviation over **all reserved observations**, after quantization, including unchanged ones |
| `required_context_features`, `min_history_hours` | Explicit measurement requirements and contiguous earlier history |
| `missing_context_policy` | `reject_window` by default; `skip_observation` leaves missing-context points exactly genuine and accounts for the return to zero offset |
| `max_candidate_attempts` | Deterministic ceiling; no relaxation to fill a budget |
| `calibration_path`, `ridge_alpha`, `residual_quantile`, `residual_floor`, `min_training_rows` | Artifact location and train-only model controls |

Unknown rainfall is **never zero-filled**. Structurally absent required features, absent
station models, insufficient history, an empty feasible band, no feasible quantum or too
little final impact produce explicit reasons. A failed attempt does not partially commit
its earlier candidate points. The bounded attempts and whole-window validation are offline
experimental procedures; causal features do not establish an online deployment guarantee.
The end-of-window return to the genuine stream is not smoothed outside the reservation.

The JSON/text summaries report model digest, required-context coverage, per-family rejection
reasons and requested/allocated/reserved/committed/unfulfilled budgets. Coverage means finite
required measurements in **planned Stealth rows**, including explicitly rejected train
windows. It does not measure overall dataset completeness, feasibility or evasion.

See [PHASE4_REPORT.md](PHASE4_REPORT.md) for the actual all-family run, calibration counts,
integrity and repeatability checks. Those are generation statistics, not detection scores.

## Export (Phase 5)

`export` is downstream of an already-executed attack plan: it never recomputes or re-runs an
attack, it only arranges and writes the results of one `AttackExecutor.execute()` call. It
writes a new, uniquely identified run directory and never overwrites an existing one.

```bash
uv run attack-engine export \
  --input ../USCRN_livraison/data/uscrn_clean.parquet \
  --config configs/dataset.yaml \
  --attack-config configs/scheduling.yaml \
  --attack-params configs/attacks.yaml \
  --output datasets/generated
```

Each run produces `datasets/generated/<run-id>/`:

```text
<run-id>/
├── uscrn_attacked.parquet   # audit only: every original column, + SOIL_MOISTURE_5_attacked,
│                            #   is_attacked, attack_type, attack_id, partition
├── detector_input.parquet  # AI detector input: keys, partition, observed (attacked) target,
│                            #   and an explicit environmental-context allowlist only
├── ground_truth.parquet    # evaluator dataset: clean_target, attacked_target, is_attacked,
│                            #   attack_type, attack_id, is_evaluable
├── attack_manifest.csv     # one row per planned window: status, capacity, realized stats
├── run_provenance.json     # input/config identities and hashes, seed, budget accounting
├── data_contract.json      # the versioned schema of every file above
├── SHA256SUMS.txt          # SHA-256 of every file in the run, including run_provenance.json
└── HANDOVER.md             # human-readable guide for the AI/evaluation team
```

Key guarantees, enforced by `attack_engine.exporting.validation` before any file is written:

- `is_attacked = 1` only if the final committed value differs from a **finite** clean target;
  a zero-effect attack or a rejected event never produces a positive label.
- `detector_input.parquet` never carries `is_attacked`, `attack_type`, `attack_id`, a second
  clean-target column, or `flag_incoherence_verticale` (derived from the clean reference).
  The context allowlist is explicit (`attack_engine.exporting.contracts`), never "every
  available column".
- Missing original target values are preserved everywhere and marked `is_evaluable = false`.
- The manifest, ground truth and `ExecutionSummary`/`BudgetLedger` totals reconcile exactly.
- Writing is atomic: everything is built in a private temporary directory and the directory is
  renamed into place only once every file is written; a failure leaves no partial run visible
  under its final name, and an existing run directory is never overwritten.
- The source Parquet (and the Stealth calibration artifact, if used) are re-hashed after
  writing and compared with their hash at load time; a mismatch fails the export.
- `run_provenance.json` never hashes itself; `SHA256SUMS.txt` is written last and covers it.

Each run directory's own `HANDOVER.md` documents the audience of each file, the detector
feature allowlist, label semantics, and the ground-truth access policy (test-partition labels
stay with the evaluator until predictions are frozen). See [PHASE5_REPORT.md](PHASE5_REPORT.md)
for the actual export run on the reference delivery.

## Scientific conventions

- **Timestamps** mark the **end** of the measured hour, in UTC. The source column is
  timezone-naive and is interpreted as UTC on an internal copy; it is never rewritten.
- `interval_start = time - 1 h`, `observation_year = year(interval_start)`. The record stamped
  `2026-01-01T00:00:00Z` therefore closes the last hour of 2025. The source `annee` column
  must equal `observation_year`.
- **Target**: `SOIL_MOISTURE_5` (m3/m3), bounds `[0.0, 0.70]` inclusive for finite values.
  Missing target values are allowed and stay missing; missing context is never treated as 0.
  The bounds are this delivery's contract, not a universal soil property.
- **Key**: `(station, time)` must be unique; each station must be hourly-continuous.
- **Partitions**, defined on the measured interval `start <= time - 1 h < end`: train
  2022–2023, validation 2024, test 2025. Partitions must not overlap and every record must
  fall in exactly one. Nothing is split or exported; partitions are validation and planning
  metadata only.
- **Window reservation** uses the half-open interval `[start, end_exclusive)` with
  `end_exclusive = last_observation_time + 1 h`. A window never crosses a partition or
  station boundary, never contains a missing target, and (for families that need it) requires
  prior observations in the same continuous segment. Windows for the same station never
  overlap (`allow_overlap=false`); windows on different stations may share the same time.
- **Determinism**: each (family, station, partition) stream draws from its own
  `Generator(PCG64(SeedSequence(...)))`, seeded from the global seed and a SHA-256 digest of
  the stable names — never Python's per-process `hash()`. The same dataset, configuration and
  dependency versions reproduce the same plan. A seed alone does not guarantee identical
  draws across NumPy versions or machines: pin versions (`uv.lock`) for bit-stable results.

## Tests and quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest

# Optional read-only check against the real delivery
ATTACK_ENGINE_REAL_DATASET=../USCRN_livraison/data/uscrn_clean.parquet uv run pytest -m real_dataset
```

The test suite uses small deterministic synthetic fixtures written to pytest temporary
directories; it never modifies the real dataset.

## Integrity

The SHA-256 of the input is computed with streaming reads and compared with the configured
expected digest. `configs/dataset.yaml` holds the digest of the known reference delivery
(`46b98115…98421`); it identifies that one file, not NOAA USCRN data in general. A missing
expectation is reported as `not_configured`, distinct from a `mismatch`.

## Current limitations

- Export (Phase 5) is downstream of Phases 1-4 only: it never recomputes an attack, trains a
  detector, implements cryptography, or exposes an ingestion API or UI. `predictions.parquet`
  (the expected AI output format) is documented in `HANDOVER.md` but not implemented here.
- Stealth's lightweight station model and in-sample residual band are an explainable local
  consistency reference, not a complete hydrological model or evidence of evasion. Climate
  changes and extrapolation can produce infeasible windows. Required context is not imputed.
- Stealth injection into train is deliberately unsupported. No quota is reallocated when
  causality, missing context or numerical/physical constraints prevent execution.
- Committed counts are the observations actually modified, which are typically fewer than the
  reserved capacity (a spike changing 1–3 points under-consumes its window), so a budget
  deficit versus the requested quota is expected and reported, not hidden.
- Replay sources are equal-length clean runs in the same cell at least `min_age_hours` earlier;
  the scheduler's `min_history_hours` remains a fixed floor, and no automatic re-scheduling is
  done to chase the target percentage after under-consumption.
- Window length is drawn uniformly in the configured bounds and capped at the remaining
  quota; it is never truncated below the minimum, so a small leftover is reported as
  unfulfilled rather than forced into an undersized window.
- Only Parquet input is supported; the whole table is loaded into memory (fine for the
  ~3.7 MB reference file).
- Physical bounds for context variables follow the delivery's cleaning rules; they are
  plausibility limits, not site-specific climatology.
- The reference-delivery profile (row count, station list) is configuration for one known
  file; other deliveries need their own profile.
- The station-identifier rule is a naming pattern; it does not consult the NOAA station list.
