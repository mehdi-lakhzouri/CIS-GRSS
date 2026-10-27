"""Run provenance and the feature contract.

Nothing here records absolute paths, host names, user names or wall-clock times:
outputs are identified by file name and SHA-256 only, so they stay anonymous and
reproducible byte-for-byte.
"""

from __future__ import annotations

import hashlib
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from signal_processing import __version__
from signal_processing.features import FEATURE_COLUMNS
from signal_processing.params import FrozenParams

_PACKAGE_DIR = Path(__file__).resolve().parent
_DEPENDENCIES = ("numpy", "pandas", "pyarrow", "pydantic", "pyyaml", "typer")

# meaning, unit, window, NaN rule, targeted attack families, per feature (d = depth in cm).
_DEPTH_TEMPLATES: dict[str, dict[str, Any]] = {
    "sp_dtheta_dry_{d}": {
        "meaning": "Hourly first difference of SOIL_MOISTURE_{d}, kept only when rain over "
        "the last 3 h (P_CALC, incl. t) < 0.5 mm, else 0. Unexplained dry-weather jumps.",
        "unit": "m3/m3 per h",
        "window": "t-1..t (rain: t-2..t)",
        "nan_rule": "NaN if SOIL_MOISTURE_{d} at t or t-1 is NaN, "
        "or any rain value in t-2..t is NaN",
        "targets": ["spike", "bias", "drift"],
    },
    "sp_innov_{d}": {
        "meaning": "Observed minus drydown prediction theta_r + (theta_prev - theta_r)*exp(-m/tau)"
        " + rain response, iterated from the last valid observation m <= 6 h earlier.",
        "unit": "m3/m3",
        "window": "last valid observation within 6 h, rain since then",
        "nan_rule": "NaN if SOIL_MOISTURE_{d} at t is NaN, no valid value in the previous 6 h "
        "(filter reset), or rain missing in the bridged hours",
        "targets": ["spike", "bias", "drift", "stealth"],
    },
    "sp_innov_z_{d}": {
        "meaning": "sp_innov_{d} / (1.4826 * train MAD of the innovation), station-specific.",
        "unit": "dimensionless",
        "window": "as sp_innov_{d}",
        "nan_rule": "NaN where sp_innov_{d} is NaN",
        "targets": ["spike", "bias", "drift", "stealth"],
    },
    "sp_cusum_{d}": {
        "meaning": "Two-sided CUSUM max(S+, S-) of sp_innov_z_{d} with train-fitted drift k_c. "
        "State is held (not updated) while 3 h rain >= 0.5 mm or rain unknown, for at most "
        "the station's train-fitted cusum_max_hold_hours per spell (P_CALC is attackable), "
        "and reset to 0 after more than 6 h without a valid z. Compare with threshold h.",
        "unit": "dimensionless",
        "window": "recursive since the last reset",
        "nan_rule": "NaN where sp_innov_z_{d} is NaN (state unchanged)",
        "targets": ["drift", "bias", "stealth"],
    },
    "sp_rough_{d}": {
        "meaning": "Trailing std (ddof=1) of the second difference of SOIL_MOISTURE_{d}: noise "
        "texture. Injected ramps/plateaus are too smooth; spikes are too rough.",
        "unit": "m3/m3 per h^2",
        "window": "24 h trailing (t-23..t), >= 18 valid second differences",
        "nan_rule": "NaN if SOIL_MOISTURE_{d} at t is NaN or fewer than 18 valid values",
        "targets": ["drift", "stealth", "spike"],
    },
}

_STATION_FEATURES: dict[str, dict[str, Any]] = {
    "sp_depth_gap": {
        "meaning": "SOIL_MOISTURE_5 - SOIL_MOISTURE_10 (vertical coherence).",
        "unit": "m3/m3",
        "window": "t",
        "nan_rule": "NaN if either depth is NaN at t",
        "targets": ["bias", "drift", "stealth", "spike"],
    },
    "sp_depth_gap_z": {
        "meaning": "(sp_depth_gap - train median) / (1.4826 * train MAD), station-specific.",
        "unit": "dimensionless",
        "window": "t",
        "nan_rule": "NaN where sp_depth_gap is NaN",
        "targets": ["bias", "drift", "stealth", "spike"],
    },
    "sp_depth_xcorr": {
        "meaning": "Max over lags L = 0..6 h of the Pearson correlation between diff(SM5) at "
        "t-L and diff(SM10) at t (10 cm lagging), over a trailing window.",
        "unit": "dimensionless [-1, 1]",
        "window": "72 h trailing, >= 48 valid pairs",
        "nan_rule": "NaN if either depth is NaN at t, fewer than 48 pairs, or zero variance",
        "targets": ["replay", "bias", "drift", "stealth"],
    },
    "sp_replay_match": {
        "meaning": "1 if the trailing 12 h window of (SM5, SM10, P_CALC), rounded to sensor "
        "precision (0.001 m3/m3, 0.1 mm), exactly equals an earlier window of the same station."
        " Windows with any NaN or a flat SM5 or SM10 are skipped (never match).",
        "unit": "flag {0, 1}",
        "window": "12 h trailing vs. all earlier windows of the station",
        "nan_rule": "Never NaN: 0 when the window is ineligible (contains NaN or is flat)",
        "targets": ["replay"],
    },
    "sp_replay_run": {
        "meaning": "Consecutive hours with sp_replay_match = 1, up to and including t.",
        "unit": "h",
        "window": "recursive",
        "nan_rule": "Never NaN: 0 where sp_replay_match = 0",
        "targets": ["replay"],
    },
}


def feature_definitions() -> dict[str, dict[str, Any]]:
    """Contract entry for every feature column, in output order."""
    expanded: dict[str, dict[str, Any]] = {}
    for template, spec in _DEPTH_TEMPLATES.items():
        for depth in ("5", "10"):
            expanded[template.format(d=depth)] = {
                key: value.format(d=depth) if isinstance(value, str) else value
                for key, value in spec.items()
            }
    expanded.update(_STATION_FEATURES)
    return {name: expanded[name] for name in FEATURE_COLUMNS}


def code_version() -> dict[str, str]:
    """Package version plus a SHA-256 over the package's source files."""
    digest = hashlib.sha256()
    for path in sorted(_PACKAGE_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return {
        "package": "signal-processing",
        "version": __version__,
        "source_sha256": digest.hexdigest(),
    }


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    versions.update({name: metadata.version(name) for name in _DEPENDENCIES})
    return versions


def build_feature_contract(params: FrozenParams, params_sha256: str) -> dict[str, Any]:
    """Human- and machine-readable description of ``signal_features.parquet``."""
    thresholds = {
        station: {
            depth: {
                "cusum_drift": dp.cusum_drift,
                "cusum_h": dp.cusum_h,
                "max_hold_hours": sp.cusum_max_hold_hours,
            }
            for depth, dp in sp.depths.items()
        }
        for station, sp in params.stations.items()
    }
    return {
        "contract_version": "1.0",
        "file": "signal_features.parquet",
        "keys": ["station", "time"],
        "row_alignment": "exactly the input rows, keys and order of detector_input.parquet",
        "time_semantics": "UTC, end of the measured hour (as in the input)",
        "causality": "a feature at time t uses only rows of the same station with time <= t",
        "fitting": "all parameters fitted on the train partition only; no labels used",
        "attack_families": ["spike", "bias", "drift", "replay", "stealth"],
        "params_sha256": params_sha256,
        "constants": params.constants.model_dump(),
        "suggested_cusum_alarm": thresholds,
        "columns": feature_definitions(),
    }


def build_run_provenance(
    *,
    input_name: str,
    input_sha256: str,
    params_name: str,
    params_sha256: str,
    params: FrozenParams,
    outputs: dict[str, str],
    input_rows: int,
    output_rows: int,
    stations: list[str],
) -> dict[str, Any]:
    """Identity of everything that went into and came out of one transform."""
    return {
        "input": {"file": input_name, "sha256": input_sha256, "rows": input_rows},
        "params": {
            "file": params_name,
            "sha256": params_sha256,
            "fitted_on_sha256": params.input_sha256,
            "train_rows": params.train_rows,
        },
        "outputs": {name: {"sha256": sha} for name, sha in outputs.items()},
        "output_rows": output_rows,
        "stations": stations,
        "feature_columns": list(FEATURE_COLUMNS),
        "code_version": code_version(),
        "package_versions": package_versions(),
    }
