from __future__ import annotations

import pytest

from attack_engine.scheduling.budget import (
    BudgetError,
    BudgetLedger,
    ReservationStatus,
    allocate_largest_remainder,
    budget_quota,
    round_half_up,
)
from attack_engine.scheduling.models import AttackFamily, ScopeCell

CELL = ScopeCell(station="AZ_Tucson_11_W", partition="test")
OTHER = ScopeCell(station="NY_Ithaca_13_E", partition="test")
SPIKE = AttackFamily.SPIKE
BIAS = AttackFamily.BIAS


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0), (0.4, 0), (0.5, 1), (1.5, 2), (2.5, 3), (9.6, 10), (12298.5, 12299)],
)
def test_round_half_up(value: float, expected: int) -> None:
    assert round_half_up(value) == expected


def test_budget_quota_zero_is_strictly_zero() -> None:
    assert budget_quota(0.0, 204983) == 0


def test_budget_quota_matches_reference_delivery() -> None:
    assert budget_quota(0.06, 204983) == 12299


@pytest.mark.parametrize("budget", [-0.1, 1.1])
def test_budget_quota_rejects_out_of_range(budget: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        budget_quota(budget, 100)


def test_largest_remainder_sums_to_total() -> None:
    allocation = allocate_largest_remainder(10, {SPIKE: 1.0, BIAS: 1.0})
    assert allocation == {SPIKE: 5, BIAS: 5}


def test_largest_remainder_distributes_remainder_deterministically() -> None:
    allocation = allocate_largest_remainder(10, {"a": 1.0, "b": 1.0, "c": 1.0})
    assert sum(allocation.values()) == 10
    # 10/3 = 3.33 each; the two largest remainders (tie) go to the first keys by name.
    assert allocation == {"a": 4, "b": 3, "c": 3}


def test_largest_remainder_proportional() -> None:
    allocation = allocate_largest_remainder(100, {"a": 3.0, "b": 1.0})
    assert allocation == {"a": 75, "b": 25}


def test_largest_remainder_zero_total() -> None:
    assert allocate_largest_remainder(0, {"a": 1.0}) == {"a": 0}


def test_largest_remainder_ignores_zero_weight_keys() -> None:
    allocation = allocate_largest_remainder(4, {"a": 1.0, "b": 0.0})
    assert allocation == {"a": 4, "b": 0}


def _ledger() -> BudgetLedger:
    allocations = {(SPIKE, CELL): 5, (BIAS, CELL): 5, (SPIKE, OTHER): 3}
    denominators = {CELL: 50, OTHER: 30}
    return BudgetLedger(allocations, denominators)


def test_reserve_reduces_remaining() -> None:
    ledger = _ledger()
    ledger.reserve("r1", SPIKE, CELL, 3)
    assert ledger.remaining(SPIKE, CELL) == 2
    assert ledger.total_reserved == 3
    assert ledger.total_committed == 0


def test_reserve_cannot_overshoot_allocation() -> None:
    ledger = _ledger()
    ledger.reserve("r1", SPIKE, CELL, 4)
    with pytest.raises(BudgetError, match="exceed the remaining budget"):
        ledger.reserve("r2", SPIKE, CELL, 2)
    assert ledger.remaining(SPIKE, CELL) == 1


def test_cancel_releases_capacity() -> None:
    ledger = _ledger()
    ledger.reserve("r1", SPIKE, CELL, 5)
    assert ledger.remaining(SPIKE, CELL) == 0
    cancelled = ledger.cancel("r1")
    assert cancelled.status is ReservationStatus.CANCELLED
    assert ledger.remaining(SPIKE, CELL) == 5
    assert ledger.total_reserved == 0


def test_commit_records_actual_and_frees_unused() -> None:
    ledger = _ledger()
    ledger.reserve("r1", SPIKE, CELL, 5)
    committed = ledger.commit("r1", 2)
    assert committed.status is ReservationStatus.COMMITTED
    assert committed.committed_count == 2
    # Reserved now reflects only what was modified; the other 3 are freed.
    cell = ledger.cell_ledger(SPIKE, CELL)
    assert (cell.reserved, cell.committed, cell.remaining) == (2, 2, 3)
    assert ledger.total_committed == 2


def test_commit_cannot_exceed_reserved() -> None:
    ledger = _ledger()
    ledger.reserve("r1", SPIKE, CELL, 3)
    with pytest.raises(BudgetError, match=r"\[0, 3\]"):
        ledger.commit("r1", 4)


def test_reserved_is_not_counted_as_modified() -> None:
    ledger = _ledger()
    ledger.reserve("r1", SPIKE, CELL, 5)
    cell = ledger.cell_ledger(SPIKE, CELL)
    assert cell.reserved == 5
    assert cell.committed == 0, "reservations are not modifications"


@pytest.mark.parametrize("operation", ["cancel", "commit"])
def test_operations_require_active_reservation(operation: str) -> None:
    ledger = _ledger()
    ledger.reserve("r1", SPIKE, CELL, 2)
    ledger.cancel("r1")
    with pytest.raises(BudgetError, match="not active"):
        ledger.cancel("r1") if operation == "cancel" else ledger.commit("r1", 1)


def test_unknown_reservation_is_rejected() -> None:
    with pytest.raises(BudgetError, match="unknown reservation"):
        _ledger().commit("missing", 1)


def test_duplicate_reservation_id_is_rejected() -> None:
    ledger = _ledger()
    ledger.reserve("r1", SPIKE, CELL, 1)
    with pytest.raises(BudgetError, match="already used"):
        ledger.reserve("r1", BIAS, CELL, 1)


def test_reserve_in_unallocated_cell_is_rejected() -> None:
    with pytest.raises(BudgetError, match="no allocation"):
        _ledger().reserve("r1", BIAS, OTHER, 1)


def test_cell_ledger_reports_denominator() -> None:
    ledger = _ledger()
    assert ledger.cell_ledger(SPIKE, CELL).finite_denominator == 50
    assert ledger.total_allocated == 13
