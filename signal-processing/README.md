# signal-processing

Turns the attack engine's **detector release** into causal, train-fitted signal
features for an AI false-data-injection detector (Track 2, in-situ soil moisture).
Independent of `attack-engine/`: its own uv environment, reads only
`detector_input.parquet`, never imports engine code.

## Guarantees

| Property | How it is enforced | Test |
|---|---|---|
| **Causal** | Per station, rows are placed on a complete hourly grid (missing hours = NaN, nothing filled). Only lags, trailing windows ending at t and forward recursions are used. | `test_causality_future_perturbation_leaves_past_unchanged` |
| **Train-only fit** | The frame is cut to the train partition before any estimate. Rule `start <= time - 1h < end`, read from `attack-engine/configs/dataset.yaml` (defaults if absent) and cross-checked against the input's `partition` column. | `test_params_ignore_validation_and_test_rows` |
| **Robust** | Huber IRLS regression, median/MAD scales, quantile thresholds. | `test_drydown_recovers_reasonable_parameters` |
| **No labels** | Label/evaluator file names are refused; only `station, time, SOIL_MOISTURE_5, SOIL_MOISTURE_10, P_CALC, partition` are read. | `test_end_to_end_without_labels`, `test_rejects_label_files_by_name` |
| **Aligned** | Output = input rows, keys and order. NaN input → NaN dependent features (replay flags are 0). | `test_alignment_*`, `test_nan_input_gives_nan_features` |
| **Gap reset** | > 6 h since the last valid value resets the drydown filter and the CUSUM. | `test_short_gap_bridged_long_gap_resets`, `test_cusum_resets_*` |
| **Deterministic** | Fixed Parquet settings, no pandas metadata, no timestamps/paths in JSON. | `test_transform_is_byte_deterministic` |
| **Anonymous** | Files are identified by name + SHA-256; machine info has no host name. | `test_end_to_end_without_labels` |

## Features (`sp_*`, d ∈ {5, 10} cm)

| Column | Meaning | Targets |
|---|---|---|
| `sp_dtheta_dry_d` | Hourly Δθ when 3 h rain < 0.5 mm, else 0 | spike, bias, drift |
| `sp_innov_d` | θ − drydown prediction `θr + (θprev − θr)·e^(−m/τ) + k·rain`, from the last valid value ≤ 6 h back | spike, bias, drift, stealth |
| `sp_innov_z_d` | innovation / (1.4826 · train MAD) | spike, bias, drift, stealth |
| `sp_cusum_d` | two-sided CUSUM `max(S+, S−)` of innov_z; suggested alarm `h` in the params/contract | drift, bias, stealth |
| `sp_depth_gap`, `sp_depth_gap_z` | SM5 − SM10, and its robust z vs. train | all but replay |
| `sp_depth_xcorr` | max Pearson r of diff(SM5) at t−L vs diff(SM10) at t, L = 0..6, 72 h trailing, ≥ 48 pairs | replay, bias, drift, stealth |
| `sp_replay_match`, `sp_replay_run` | exact repeat of the trailing **12 h** (SM5, SM10, P) window at sensor precision among earlier windows; run length | replay |
| `sp_rough_d` | 24 h trailing std of the second difference | drift, stealth, spike |

Full definitions (unit, window, NaN rule) are written to `feature_contract.json`.

### Design decisions (deviations from the first spec, with evidence)

- **Replay window = 12 h** (not 24 h). Replay events last 6–72 h. On the clean
  delivery, exact repeats of eligible windows: 6 h → 37, 12 h → 0, 24 h → 0. 12 h
  catches replays ≥ 12 h with zero natural matches (train: 0 matches at every station).
  Windows with a NaN, or a flat SM5 or SM10, are never eligible.
- **CUSUM drift `k_c` fitted per station/depth:** `max(0.5, q90 of |24 h trailing mean
  of z|)` on dry train hours. With a fixed 0.5, persistent model bias (dry-period
  median z ≈ 0.55 at one station) made the train statistic run to > 1,300.
- **CUSUM held during rain, with a cap.** Rain makes innov_z explode, so the state is
  held while 3 h rain ≥ 0.5 mm. P_CALC is itself attacked: a drift event was seen
  ramping fake rain over 64 h to freeze the CUSUM. The hold is therefore capped per
  station at `ceil(q99 of train hold-run lengths)`, clipped to [6, 48] h.
- **Drydown fit** uses the equivalent linear form `θt = c + a·θt−1 + k·Pt` (a = e^(−1/τ),
  θr = c/(1−a)) with Huber IRLS, k ≥ 0 and τ ∈ [1, 10⁴] h. Scales are floored at
  0.0005 m³/m³ (half the sensor resolution) so MAD = 0 cannot divide by zero.

## Reproduce (Windows PowerShell; same commands in bash)

uv is required (`pip install --user uv`, or the official installer).

```powershell
# 1. Export the combined campaign (from CIS-GRSS\attack-engine)
uv sync --locked
uv run attack-engine export --input ../USCRN_livraison/data/uscrn_clean.parquet --config configs/dataset.yaml --attack-config configs/campaigns/combined/scheduling.yaml --attack-params configs/campaigns/combined/attacks.yaml --output datasets/generated

# 2. Features (from CIS-GRSS\signal-processing); set $D to the new run's detector folder
$D = (Get-ChildItem ../attack-engine/datasets/generated -Directory | Sort-Object Name | Select-Object -Last 1).FullName + "/detector"
uv sync --all-groups
uv run signal-processing fit --input $D/detector_input.parquet --output params/frozen_params.json
uv run signal-processing transform --input $D/detector_input.parquet --params params/frozen_params.json --output outputs/combined
uv run signal-processing bench --input $D/detector_input.parquet --params params/frozen_params.json --output outputs/bench.json

# 3. Evaluation evidence (the ONLY step that reads labels; validation partition only)
uv run python scripts/feature_report.py --features outputs/combined/signal_features.parquet --labels $D/row_labels.parquet --input $D/detector_input.parquet --params params/frozen_params.json --output-dir outputs/report

# 4. Checks
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

`fit` refuses to overwrite a params file with different content (exit 1); delete it
deliberately to refit. `params/frozen_params.json` is committed and records the
SHA-256 of the input it was fitted on plus the train boundaries.

## Outputs for the detector team

`outputs/combined/`:

- `signal_features.parquet` — `station`, `time` + 15 `sp_*` columns, aligned 1:1 with `detector_input.parquet`
- `feature_contract.json` — per-column meaning, unit, window, NaN rule, targeted families; constants; CUSUM thresholds
- `run_provenance.json` — input/params/output SHA-256, code version (source hash), package versions, row counts

## Layout

```text
src/signal_processing/
  io.py          reading the release, deterministic writing, hashing
  partition.py   engine partition rule
  params.py      frozen-parameter schema (pydantic)
  fit.py         train-only robust fitting
  features.py    causal feature computation
  provenance.py  feature contract and run provenance
  bench.py       latency / throughput / memory
  cli.py         fit | transform | bench
scripts/feature_report.py   label-reading evaluation (validation only)
params/frozen_params.json   committed frozen params
tests/                      pytest suite (synthetic data)
```
