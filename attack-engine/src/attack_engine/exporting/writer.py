"""Atomic, fail-safe writing of one export run directory.

Every file is written to a private temporary directory first; the directory is
only renamed to its final, public name once every file has been written and
verified, so a crash or an exception never leaves a partial run that looks
complete. Existing runs are never overwritten.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pandas as pd

from attack_engine.exporting.models import ExportError
from attack_engine.utils.hashing import sha256_file

SHA256SUMS_FILE = "SHA256SUMS.txt"


def reserve_run_directory(output_root: Path, run_id: str) -> tuple[Path, Path]:
    """Create a private temp directory for ``run_id`` under ``output_root``.

    Returns ``(temp_dir, final_dir)``. Raises :class:`ExportError` if the final
    directory already exists, before anything is written.
    """
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"cannot create output directory {output_root}: {exc}") from exc
    final_dir = output_root / run_id
    if final_dir.resolve().parent != output_root.resolve():
        raise ExportError("run directory must stay within the output root")
    if final_dir.exists():
        raise ExportError(f"run directory already exists, refusing to overwrite: {final_dir}")
    temp_dir = output_root / f".tmp-{run_id}-{uuid4().hex[:8]}"
    try:
        temp_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise ExportError(f"cannot create temporary run directory {temp_dir}: {exc}") from exc
    return temp_dir, final_dir


def finalize_run_directory(temp_dir: Path, final_dir: Path) -> None:
    """Atomically publish a fully written temp directory as the final run directory."""
    if final_dir.exists():
        raise ExportError(f"run directory already exists, refusing to overwrite: {final_dir}")
    try:
        temp_dir.rename(final_dir)
    except OSError as exc:
        raise ExportError(f"cannot finalize run directory {final_dir}: {exc}") from exc


def discard_run_directory(temp_dir: Path) -> None:
    # Only remove a private staging directory created directly under the chosen output root.
    resolved = temp_dir.resolve()
    if not resolved.name.startswith(".tmp-") or resolved.parent != temp_dir.parent.resolve():
        raise ExportError(f"refusing cleanup outside the private staging directory: {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    try:
        frame.to_parquet(path, index=False)
    except OSError as exc:
        raise ExportError(f"cannot write {path}: {exc}") from exc


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    try:
        frame.to_csv(path, index=False)
    except OSError as exc:
        raise ExportError(f"cannot write {path}: {exc}") from exc


def write_json(path: Path, document: dict) -> None:
    try:
        path.write_text(json.dumps(document, indent=2, sort_keys=False), encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"cannot write {path}: {exc}") from exc


def write_text(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"cannot write {path}: {exc}") from exc


def hash_files(directory: Path, file_names: list[str]) -> dict[str, str]:
    """SHA-256 of each named file, read back from disk (not from in-memory buffers)."""
    return {name: sha256_file(directory / name) for name in file_names}


def write_sha256sums(directory: Path, hashes: dict[str, str]) -> None:
    lines = [f"{digest}  {name}" for name, digest in sorted(hashes.items())]
    write_text(directory / SHA256SUMS_FILE, "\n".join(lines) + "\n")


def run_in_temp_directory(
    output_root: Path, run_id: str, write_all: Callable[[Path], None]
) -> Path:
    """Write everything under a temp directory, then atomically publish it.

    ``write_all`` receives the temp directory and must write every artifact
    there; any exception discards the temp directory entirely, so no partial
    run is ever visible under its final name.
    """
    temp_dir, final_dir = reserve_run_directory(output_root, run_id)
    try:
        write_all(temp_dir)
        finalize_run_directory(temp_dir, final_dir)
    except BaseException:
        discard_run_directory(temp_dir)
        raise
    return final_dir
