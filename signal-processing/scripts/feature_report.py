"""Evaluation-only report: per-family ROC-AUC of each signal feature on VALIDATION.

This is the ONLY code in the module that reads labels (``row_labels.parquet``).
It is evidence for the report, not a detector: no threshold or model is fitted
here, and nothing it computes flows back into ``fit`` or ``transform``.

Score = |feature| (direction-free). Positives: validation rows whose
``attack_types`` contain the family. Negatives: validation rows with
``is_attacked`` false. Rows where the feature is NaN are dropped and the
coverage of positives is reported. AUC is the Mann-Whitney statistic.

Usage (from signal-processing/, needs ``uv sync --group report``):
    uv run python scripts/feature_report.py --features outputs/combined/signal_features.parquet
        --labels <detector>/row_labels.parquet --input <detector>/detector_input.parquet
        --params params/frozen_params.json --output-dir outputs/report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

from signal_processing.features import FEATURE_COLUMNS
from signal_processing.io import read_detector_input
from signal_processing.params import FrozenParams

FAMILIES = ("spike", "bias", "drift", "replay", "stealth")
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"5": "#2a78d6", "10": "#eb6834"}  # validated categorical slots 1-2


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based ROC-AUC (ties get average ranks)."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_validation(features: Path, labels: Path) -> pd.DataFrame:
    feats = pd.read_parquet(features)
    lab = pd.read_parquet(
        labels, columns=["station", "time", "partition", "is_attacked", "attack_types"]
    )
    merged = feats.merge(lab, on=["station", "time"], how="inner", validate="one_to_one")
    return merged[merged["partition"] == "validation"].reset_index(drop=True)


def auc_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows: features. Columns: AUC per family and 'any', plus positive coverage."""
    attacked = frame["is_attacked"].to_numpy(bool)
    types = frame["attack_types"].apply(lambda v: set(v) if v is not None else set())
    targets = {f: types.apply(lambda s, f=f: f in s).to_numpy(bool) for f in FAMILIES}
    targets["any"] = attacked
    rows = []
    for column in FEATURE_COLUMNS:
        score = np.abs(frame[column].to_numpy(float))
        finite = np.isfinite(score)
        row: dict[str, float | str] = {"feature": column}
        for name, pos in targets.items():
            keep = finite & (pos | ~attacked)
            row[name] = auc(score[keep], pos[keep])
        row["pos_coverage"] = float(finite[attacked].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def counts(frame: pd.DataFrame) -> dict[str, int]:
    types = frame["attack_types"].apply(lambda v: set(v) if v is not None else set())
    out = {f: int(types.apply(lambda s, f=f: f in s).sum()) for f in FAMILIES}
    out["negatives"] = int((~frame["is_attacked"].astype(bool)).sum())
    return out


def pick_event(labels: Path) -> tuple[str, pd.Timestamp, pd.Timestamp, str]:
    """Longest single-family drift event in validation (deterministic choice)."""
    lab = pd.read_parquet(
        labels, columns=["station", "time", "partition", "attack_type", "attack_id"]
    )
    lab = lab[(lab["partition"] == "validation") & (lab["attack_type"] == "drift")]
    spans = lab.groupby("attack_id").agg(
        station=("station", "first"), start=("time", "min"), end=("time", "max"), n=("time", "size")
    )
    best = spans.sort_values(["n", "start"], ascending=[False, True]).iloc[0]
    return best["station"], best["start"], best["end"], str(best.name)


def plot_event(
    detector_input: Path, features: Path, labels: Path, params: FrozenParams, out: Path
) -> str:
    """Raw signal, innovation z and CUSUM around one attacked window (3 aligned panels)."""
    station, start, end, attack_id = pick_event(labels)
    lo, hi = start - pd.Timedelta(days=4), end + pd.Timedelta(days=4)
    raw = read_detector_input(detector_input)
    feats = pd.read_parquet(features)
    window = (raw["station"] == station) & raw["time"].between(lo, hi)
    raw, feats = raw[window], feats[window.to_numpy()]
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, facecolor=SURFACE)
    panels = (
        ("Soil moisture (m³/m³)", "SOIL_MOISTURE_{d}", raw),
        ("Innovation z", "sp_innov_z_{d}", feats),
        ("CUSUM statistic", "sp_cusum_{d}", feats),
    )
    for ax, (ylabel, template, source) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        ax.axvspan(start - pd.Timedelta(hours=1), end, color=GRID, zorder=0, lw=0)
        for depth, color in SERIES.items():
            ax.plot(
                source["time"],
                source[template.format(d=depth)],
                color=color,
                lw=1.6,
                label=f"{depth} cm",
            )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for depth, color in SERIES.items():
        h = params.stations[station].depths[depth].cusum_h
        axes[2].axhline(h, color=color, lw=1, ls=(0, (4, 3)))
        right = depth == "10"
        axes[2].annotate(
            f"h ({depth} cm) = {h:.0f}",
            (1.0 if right else 0.0, h),
            xycoords=("axes fraction", "data"),
            ha="right" if right else "left",
            va="bottom",
            color=INK_MUTED,
            fontsize=8,
        )
    axes[0].legend(frameon=False, loc="upper left", ncols=2, labelcolor=INK)
    axes[0].text(
        start,
        axes[0].get_ylim()[1],
        " attacked window (label)",
        va="top",
        color=INK_MUTED,
        fontsize=8,
    )
    axes[2].set_yscale("symlog", linthresh=10)
    axes[2].set_ylim(bottom=0)
    fig.suptitle(f"{station}: validation drift event {attack_id}", color=INK, x=0.06, ha="left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return f"{station} {start} -> {end} ({attack_id})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="detector_input.parquet")
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_validation(args.features, args.labels)
    table = auc_table(frame)
    n = counts(frame)
    header = "Validation rows: " + ", ".join(f"{k}={v}" for k, v in n.items())
    markdown = table.to_markdown(floatfmt=".3f") if _has_tabulate() else table.round(3).to_string()
    (args.output_dir / "feature_auc.md").write_bytes(f"{header}\n\n{markdown}\n".encode())
    table.to_csv(args.output_dir / "feature_auc.csv", float_format="%.4f", lineterminator="\n")
    print(header)
    print(table.round(3).to_string())

    params = FrozenParams.model_validate_json(args.params.read_bytes())
    event = plot_event(
        args.input, args.features, args.labels, params, args.output_dir / "attacked_window.png"
    )
    print(f"figure: {args.output_dir / 'attacked_window.png'} [{event}]")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
