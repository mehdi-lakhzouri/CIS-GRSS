"""Streaming SHA-256 hashing and integrity comparison for source files.

Files are only ever opened for reading, in fixed-size chunks, so large datasets
are never loaded into memory just to be hashed.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DEFAULT_CHUNK_SIZE = 1024 * 1024
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class IntegrityStatus(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    NOT_CONFIGURED = "not_configured"


class IntegrityResult(BaseModel):
    """Outcome of comparing a computed digest with an optional expected digest."""

    model_config = ConfigDict(frozen=True)

    actual_sha256: str
    expected_sha256: str | None
    status: IntegrityStatus

    @property
    def ok(self) -> bool:
        """False only for a mismatch; an absent expectation is not a failure."""
        return self.status is not IntegrityStatus.MISMATCH


def sha256_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the lowercase hex SHA-256 digest of ``path`` using streaming reads."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sha256(value: str) -> str:
    """Return ``value`` as a lowercase 64-character hex digest, or raise ``ValueError``."""
    normalized = value.strip().lower()
    if not _SHA256_HEX.fullmatch(normalized):
        raise ValueError(f"not a valid SHA-256 hex digest: {value!r}")
    return normalized


def compare_sha256(actual: str, expected: str | None) -> IntegrityResult:
    """Compare a computed digest with an expected one, distinguishing a missing expectation."""
    actual = normalize_sha256(actual)
    if expected is None:
        return IntegrityResult(
            actual_sha256=actual, expected_sha256=None, status=IntegrityStatus.NOT_CONFIGURED
        )
    expected = normalize_sha256(expected)
    status = IntegrityStatus.MATCH if actual == expected else IntegrityStatus.MISMATCH
    return IntegrityResult(actual_sha256=actual, expected_sha256=expected, status=status)
