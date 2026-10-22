from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from attack_engine.utils.hashing import IntegrityStatus, compare_sha256, sha256_file

# FIPS 180-2 test vectors.
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.mark.parametrize(("content", "digest"), [(b"abc", SHA256_ABC), (b"", SHA256_EMPTY)])
def test_known_digests(tmp_path: Path, content: bytes, digest: str) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(content)
    assert sha256_file(path) == digest


@pytest.mark.parametrize("chunk_size", [1, 7, 4096, 10**7])
def test_digest_is_independent_of_chunk_size(tmp_path: Path, chunk_size: int) -> None:
    content = bytes(range(256)) * 1000
    path = tmp_path / "data.bin"
    path.write_bytes(content)
    assert sha256_file(path, chunk_size) == hashlib.sha256(content).hexdigest()


def test_hashing_does_not_modify_file(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"payload")
    mtime = path.stat().st_mtime_ns
    sha256_file(path)
    assert path.read_bytes() == b"payload"
    assert path.stat().st_mtime_ns == mtime


def test_invalid_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="positive"):
        sha256_file(path, 0)


def test_match_is_case_insensitive() -> None:
    result = compare_sha256(SHA256_ABC, SHA256_ABC.upper())
    assert result.status is IntegrityStatus.MATCH
    assert result.ok


def test_mismatch_is_detected() -> None:
    result = compare_sha256(SHA256_ABC, SHA256_EMPTY)
    assert result.status is IntegrityStatus.MISMATCH
    assert not result.ok
    assert result.actual_sha256 == SHA256_ABC
    assert result.expected_sha256 == SHA256_EMPTY


def test_missing_expectation_is_distinct_from_mismatch() -> None:
    result = compare_sha256(SHA256_ABC, None)
    assert result.status is IntegrityStatus.NOT_CONFIGURED
    assert result.ok
    assert result.expected_sha256 is None


@pytest.mark.parametrize("expected", ["abc", "z" * 64, SHA256_ABC + "0"])
def test_malformed_expected_hash_is_rejected(expected: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        compare_sha256(SHA256_ABC, expected)
