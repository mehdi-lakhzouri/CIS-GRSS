"""Deterministic seed derivation for reproducible scheduling.

Random streams are tied to *stable names* (station, partition, attack family)
rather than to their position in a list, so adding a family or reordering
stations does not shift the draws of the others.

Seeds are derived from a SHA-256 digest of a canonical name joined with the
global seed through :class:`numpy.random.SeedSequence`. Python's built-in
``hash()`` is deliberately avoided: it is salted per process and would make runs
irreproducible.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Separator that cannot appear in a station id, partition name or family name,
# so distinct name tuples can never collapse to the same canonical string.
_NAME_SEPARATOR = "\x1f"
_DIGEST_BITS = 128


def canonical_name(*parts: str) -> str:
    """Join name parts into one unambiguous canonical string."""
    for part in parts:
        if _NAME_SEPARATOR in part:
            raise ValueError(f"name part {part!r} contains the reserved separator")
    return _NAME_SEPARATOR.join(parts)


def _digest_entropy(name: str) -> int:
    """Return a stable integer derived from the SHA-256 digest of ``name``."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[: _DIGEST_BITS // 8], byteorder="big")


def derive_seed_sequence(global_seed: int, *parts: str) -> np.random.SeedSequence:
    """Derive a child :class:`~numpy.random.SeedSequence` from a global seed and a name.

    The same global seed and name parts always yield the same sequence, within a
    fixed NumPy version (see the module docstring of :mod:`attack_engine.scheduling`).
    """
    entropy = [int(global_seed), _digest_entropy(canonical_name(*parts))]
    return np.random.SeedSequence(entropy)


def derive_generator(global_seed: int, *parts: str) -> np.random.Generator:
    """Return a PCG64 :class:`~numpy.random.Generator` seeded deterministically."""
    return np.random.Generator(np.random.PCG64(derive_seed_sequence(global_seed, *parts)))
