"""Chronological partitions, using the attack engine's rule.

A row belongs to a partition when ``start <= time - 1 h < end`` (UTC): timestamps
mark the END of the measured hour. Bounds are read from the engine's
``configs/dataset.yaml`` when available, otherwise the defaults below are used.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, field_validator

RULE_TEXT = "start <= time - 1h < end (UTC, time marks the end of the measured hour)"

# Repository-relative location of the engine configuration (recorded as-is, never absolute).
ENGINE_CONFIG_RELATIVE = "attack-engine/configs/dataset.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[3]


class PartitionRule(BaseModel):
    """One named, half-open interval on the measured hour."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    def hour_bounds(self) -> tuple[int, int]:
        """``(start, end)`` as whole hours since the epoch."""
        return int(self.start.timestamp()) // 3600, int(self.end.timestamp()) // 3600


DEFAULT_RULES = (
    PartitionRule(
        name="train", start=datetime(2022, 1, 1, tzinfo=UTC), end=datetime(2024, 1, 1, tzinfo=UTC)
    ),
    PartitionRule(
        name="validation",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, tzinfo=UTC),
    ),
    PartitionRule(
        name="test", start=datetime(2025, 1, 1, tzinfo=UTC), end=datetime(2026, 1, 1, tzinfo=UTC)
    ),
)


def load_partition_rules(config_path: Path | None = None) -> tuple[tuple[PartitionRule, ...], str]:
    """Return the rules and a description of their (non-absolute) source.

    With no explicit path, the engine config next to this package is used if it
    exists; otherwise the built-in defaults (identical to the engine's values).
    """
    if config_path is None:
        candidate = _REPO_ROOT / ENGINE_CONFIG_RELATIVE
        if not candidate.is_file():
            return DEFAULT_RULES, "built-in defaults"
        config_path, source = candidate, ENGINE_CONFIG_RELATIVE
    else:
        source = config_path.name
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rules = tuple(PartitionRule(**entry) for entry in document["partitions"])
    if "train" not in {rule.name for rule in rules}:
        raise ValueError(f"{source} defines no 'train' partition")
    return rules, source


def assign_partitions(hours: np.ndarray, rules: tuple[PartitionRule, ...]) -> np.ndarray:
    """Partition name per row (``""`` when no rule matches)."""
    measured = hours - 1
    out = np.full(hours.shape, "", dtype=object)
    for rule in rules:
        start, end = rule.hour_bounds()
        out[(measured >= start) & (measured < end)] = rule.name
    return out


def train_mask(hours: np.ndarray, rules: tuple[PartitionRule, ...]) -> np.ndarray:
    """Boolean mask of rows in the train partition."""
    return assign_partitions(hours, rules) == "train"
