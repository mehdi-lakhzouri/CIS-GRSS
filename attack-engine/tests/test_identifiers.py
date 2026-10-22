from __future__ import annotations

import numpy as np
import pytest

from attack_engine.scheduling import identifiers


def test_same_name_and_seed_are_reproducible() -> None:
    first = identifiers.derive_generator(14, "spike", "AZ_Tucson_11_W", "test")
    second = identifiers.derive_generator(14, "spike", "AZ_Tucson_11_W", "test")
    assert (
        first.integers(0, 1_000_000, size=20).tolist()
        == second.integers(0, 1_000_000, size=20).tolist()
    )


def test_different_names_give_independent_streams() -> None:
    a = identifiers.derive_generator(14, "spike", "AZ_Tucson_11_W", "test")
    b = identifiers.derive_generator(14, "bias", "AZ_Tucson_11_W", "test")
    c = identifiers.derive_generator(14, "spike", "NY_Ithaca_13_E", "test")
    assert a.integers(0, 1_000_000, size=20).tolist() != b.integers(0, 1_000_000, size=20).tolist()
    assert a.integers(0, 1_000_000, size=20).tolist() != c.integers(0, 1_000_000, size=20).tolist()


def test_different_seed_changes_stream() -> None:
    a = identifiers.derive_generator(14, "spike", "AZ_Tucson_11_W", "test")
    b = identifiers.derive_generator(15, "spike", "AZ_Tucson_11_W", "test")
    assert a.integers(0, 1_000_000, size=20).tolist() != b.integers(0, 1_000_000, size=20).tolist()


def test_name_order_matters() -> None:
    a = identifiers.derive_seed_sequence(14, "a", "b")
    b = identifiers.derive_seed_sequence(14, "b", "a")
    assert a.generate_state(4).tolist() != b.generate_state(4).tolist()


def test_reserved_separator_is_rejected() -> None:
    with pytest.raises(ValueError, match="separator"):
        identifiers.canonical_name("station\x1fwith-separator")


def test_generator_is_pcg64() -> None:
    generator = identifiers.derive_generator(0, "x")
    assert isinstance(generator.bit_generator, np.random.PCG64)
