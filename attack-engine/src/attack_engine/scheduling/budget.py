"""Budget computation and a safe reservation ledger.

Accounting separates five quantities that must never be conflated:

- **requested**  - the budget fraction and the absolute quota it implies;
- **allocated**  - the integer quota handed to each (family, station, partition)
  cell by the largest-remainder method;
- **reserved**   - capacity held by scheduled windows; *not* a count of attacked
  observations;
- **committed**  - observations actually modified, reported by Phase 3 later;
- **remaining / unfulfilled** - allocation not yet reserved, with reasons.

The ledger never lets reservations exceed a cell's allocation, so the configured
budget cannot be silently overshot. Cancelling or releasing a reservation frees
its capacity again.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from attack_engine.scheduling.models import AttackFamily, ScopeCell


def round_half_up(value: float) -> int:
    """Round to the nearest integer, halves going up. Exact ``0`` stays ``0``."""
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def budget_quota(budget: float, denominator: int) -> int:
    """Absolute quota ``round_half_up(budget * denominator)``; a zero budget gives zero."""
    if not 0.0 <= budget <= 1.0:
        raise ValueError(f"budget must be in [0, 1], got {budget}")
    if denominator < 0:
        raise ValueError(f"denominator must be non-negative, got {denominator}")
    if budget == 0.0:
        return 0
    return round_half_up(budget * denominator)


def allocate_largest_remainder(total: int, weights: Mapping[object, float]) -> dict[object, int]:
    """Split ``total`` integer units across keys proportionally to ``weights``.

    Uses the largest-remainder (Hamilton) method so the parts sum exactly to
    ``total``. Ties in the fractional remainder are broken by the key's string
    form, keeping the result deterministic.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    positive = {key: float(weight) for key, weight in weights.items() if weight > 0.0}
    if not positive or total == 0:
        return dict.fromkeys(weights, 0)

    weight_sum = sum(positive.values())
    exact = {key: total * weight / weight_sum for key, weight in positive.items()}
    floors = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(floors.values())
    order = sorted(positive, key=lambda key: (-(exact[key] - floors[key]), str(key)))
    for key in order[:remainder]:
        floors[key] += 1
    return {key: floors.get(key, 0) for key in weights}


class ReservationStatus(StrEnum):
    ACTIVE = "active"
    COMMITTED = "committed"
    CANCELLED = "cancelled"


class Reservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    reservation_id: str
    family: AttackFamily
    cell: ScopeCell
    reserved_count: int = Field(ge=1)
    status: ReservationStatus = ReservationStatus.ACTIVE
    committed_count: int | None = None


class CellLedger(BaseModel):
    """Accounting for one (family, station, partition) cell."""

    model_config = ConfigDict(frozen=True)

    family: AttackFamily
    cell: ScopeCell
    finite_denominator: int
    allocated: int
    reserved: int
    committed: int
    active_reservation_count: int

    @property
    def remaining(self) -> int:
        """Allocation not yet held by an active reservation."""
        return self.allocated - self.reserved

    @property
    def unfulfilled(self) -> int:
        return self.remaining


class BudgetError(Exception):
    """An invalid ledger operation (overshoot, unknown or reused reservation id)."""


class BudgetLedger:
    """Mutable reservation ledger over a fixed set of cell allocations.

    Scheduling calls :meth:`reserve` / :meth:`cancel`; Phase 3 will call
    :meth:`commit` with the number of observations actually modified. Reserved
    capacity and committed counts are tracked separately and a commit can never
    exceed what was reserved.
    """

    def __init__(
        self,
        allocations: Mapping[tuple[AttackFamily, ScopeCell], int],
        denominators: Mapping[ScopeCell, int],
    ) -> None:
        self._allocated = dict(allocations)
        self._denominators = dict(denominators)
        self._reserved: dict[tuple[AttackFamily, ScopeCell], int] = dict.fromkeys(allocations, 0)
        self._committed: dict[tuple[AttackFamily, ScopeCell], int] = dict.fromkeys(allocations, 0)
        self._active_count: dict[tuple[AttackFamily, ScopeCell], int] = dict.fromkeys(
            allocations, 0
        )
        self._reservations: dict[str, Reservation] = {}

    def allocated(self, family: AttackFamily, cell: ScopeCell) -> int:
        return self._allocated.get((family, cell), 0)

    def remaining(self, family: AttackFamily, cell: ScopeCell) -> int:
        key = (family, cell)
        return self._allocated.get(key, 0) - self._reserved.get(key, 0)

    def reserve(
        self, reservation_id: str, family: AttackFamily, cell: ScopeCell, count: int
    ) -> Reservation:
        """Hold ``count`` units in a cell. Refuses to exceed the cell's allocation."""
        if count < 1:
            raise BudgetError(f"reservation count must be positive, got {count}")
        if reservation_id in self._reservations:
            raise BudgetError(f"reservation id already used: {reservation_id!r}")
        key = (family, cell)
        if key not in self._allocated:
            raise BudgetError(f"no allocation for {family.value} in {cell.label()}")
        if count > self.remaining(family, cell):
            raise BudgetError(
                f"reserving {count} would exceed the remaining budget "
                f"{self.remaining(family, cell)} for {family.value} in {cell.label()}"
            )
        self._reserved[key] += count
        self._active_count[key] += 1
        reservation = Reservation(
            reservation_id=reservation_id, family=family, cell=cell, reserved_count=count
        )
        self._reservations[reservation_id] = reservation
        return reservation

    def cancel(self, reservation_id: str) -> Reservation:
        """Release an active reservation, freeing its capacity."""
        reservation = self._require_active(reservation_id)
        key = (reservation.family, reservation.cell)
        self._reserved[key] -= reservation.reserved_count
        self._active_count[key] -= 1
        updated = reservation.model_copy(update={"status": ReservationStatus.CANCELLED})
        self._reservations[reservation_id] = updated
        return updated

    def commit(self, reservation_id: str, modified_count: int) -> Reservation:
        """Record how many observations an active reservation actually modified.

        For Phase 3: ``modified_count`` may be fewer than the reserved count (an
        attack can change fewer positions than it reserved) but never more.
        """
        reservation = self._require_active(reservation_id)
        if not 0 <= modified_count <= reservation.reserved_count:
            raise BudgetError(
                f"committed count {modified_count} must be within "
                f"[0, {reservation.reserved_count}] for {reservation_id!r}"
            )
        key = (reservation.family, reservation.cell)
        # Free the unused part of the reservation; keep only what was modified.
        self._reserved[key] -= reservation.reserved_count - modified_count
        self._committed[key] += modified_count
        self._active_count[key] -= 1
        updated = reservation.model_copy(
            update={"status": ReservationStatus.COMMITTED, "committed_count": modified_count}
        )
        self._reservations[reservation_id] = updated
        return updated

    def _require_active(self, reservation_id: str) -> Reservation:
        try:
            reservation = self._reservations[reservation_id]
        except KeyError:
            raise BudgetError(f"unknown reservation id: {reservation_id!r}") from None
        if reservation.status is not ReservationStatus.ACTIVE:
            raise BudgetError(
                f"reservation {reservation_id!r} is {reservation.status.value}, not active"
            )
        return reservation

    def reservation(self, reservation_id: str) -> Reservation:
        return self._reservations[reservation_id]

    def cell_ledger(self, family: AttackFamily, cell: ScopeCell) -> CellLedger:
        key = (family, cell)
        return CellLedger(
            family=family,
            cell=cell,
            finite_denominator=self._denominators.get(cell, 0),
            allocated=self._allocated.get(key, 0),
            reserved=self._reserved.get(key, 0),
            committed=self._committed.get(key, 0),
            active_reservation_count=self._active_count.get(key, 0),
        )

    def cell_ledgers(self) -> list[CellLedger]:
        return [self.cell_ledger(family, cell) for family, cell in self._allocated]

    @property
    def total_allocated(self) -> int:
        return sum(self._allocated.values())

    @property
    def total_reserved(self) -> int:
        return sum(self._reserved.values())

    @property
    def total_committed(self) -> int:
        return sum(self._committed.values())
