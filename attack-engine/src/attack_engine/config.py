"""Project configuration: YAML files parsed into validated, immutable models."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from attack_engine.dataset.schema import PartitionPlan, ValidationProfile
from attack_engine.utils.hashing import normalize_sha256


class ConfigError(Exception):
    """The configuration file is missing, unreadable or invalid."""


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # SHA-256 of the known reference delivery; ``None`` disables the comparison.
    expected_sha256: str | None = None
    validation_profile: ValidationProfile
    partitions: PartitionPlan | None = None

    @field_validator("expected_sha256")
    @classmethod
    def _normalize_hash(cls, value: str | None) -> str | None:
        return None if value is None else normalize_sha256(value)


def load_config(path: Path) -> AppConfig:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {path}") from None
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"configuration {path} must contain a mapping at the top level")
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration {path}:\n{exc}") from exc
