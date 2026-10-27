"""Transform latency, throughput and peak memory on a given input."""

from __future__ import annotations

import os
import platform
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from signal_processing.features import transform
from signal_processing.params import FrozenParams
from signal_processing.provenance import code_version, package_versions


def machine_info() -> dict[str, Any]:
    """Hardware/OS description without host or user names."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def run_bench(frame: pd.DataFrame, params: FrozenParams, repeats: int = 5) -> dict[str, Any]:
    """Time the transform per station chunk, repeated; then one traced full run.

    Per-row latency = chunk wall time / chunk rows (the transform is batched per
    station, so this is the amortised cost of one row).
    """
    stations = sorted(frame["station"].unique())
    chunks = [frame[frame["station"] == s].reset_index(drop=True) for s in stations]
    transform(chunks[0], params)  # warm-up
    per_row_us: list[float] = []
    full_seconds: list[float] = []
    for _ in range(repeats):
        for chunk in chunks:
            start = time.perf_counter()
            transform(chunk, params)
            per_row_us.append((time.perf_counter() - start) * 1e6 / len(chunk))
        start = time.perf_counter()
        transform(frame, params)
        full_seconds.append(time.perf_counter() - start)
    tracemalloc.start()
    transform(frame, params)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    best = float(np.median(full_seconds))
    return {
        "rows": len(frame),
        "stations": len(stations),
        "repeats": repeats,
        "per_row_us": {
            "median": float(np.median(per_row_us)),
            "p95": float(np.quantile(per_row_us, 0.95)),
            "basis": "station chunk wall time / chunk rows",
        },
        "full_transform_seconds_median": best,
        "rows_per_second": len(frame) / best,
        "peak_memory_mib_tracemalloc": peak / 2**20,
        "machine": machine_info(),
        "package_versions": package_versions(),
        "code_version": code_version(),
    }
