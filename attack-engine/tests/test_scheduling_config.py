from __future__ import annotations

from pathlib import Path

import pytest

from attack_engine.scheduling.config import (
    ConfigError,
    InsufficientPolicy,
    SchedulingConfig,
    load_scheduling_config,
)
from attack_engine.scheduling.models import AttackFamily

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "scheduling.yaml"

_VALID = """
global_seed: 14
attack_budget: 0.06
budget_scope: [test]
enabled_families: [spike, bias]
min_window_length: 6
max_window_length: 168
max_attempts: 1000
family_parameters:
  spike: {weight: 0.5, min_window: 1, max_window: 6}
  bias: {weight: 0.5}
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "scheduling.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_project_config_loads() -> None:
    config = load_scheduling_config(PROJECT_CONFIG)
    assert config.global_seed == 14
    assert config.attack_budget == 0.06
    assert set(config.enabled_families) == set(AttackFamily)
    assert config.allow_overlap is False
    assert config.insufficient_policy is InsufficientPolicy.REPORT
    assert config.window_bounds(AttackFamily.SPIKE) == (1, 6)
    assert config.window_bounds(AttackFamily.BIAS) == (6, 168)


def test_valid_minimal_config(tmp_path: Path) -> None:
    config = load_scheduling_config(_write(tmp_path, _VALID))
    assert config.budget_scope == ("test",)
    assert config.window_bounds(AttackFamily.SPIKE) == (1, 6)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("attack_budget: 0.06", "attack_budget: -0.1", "greater than or equal"),
        ("attack_budget: 0.06", "attack_budget: 1.5", "less than or equal"),
        ("max_attempts: 1000", "max_attempts: 0", "greater than 0"),
        ("max_window_length: 168", "max_window_length: 2", "min_window_length"),
    ],
)
def test_invalid_scalar_fields_are_rejected(tmp_path: Path, old: str, new: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        load_scheduling_config(_write(tmp_path, _VALID.replace(old, new)))


def test_empty_family_list_is_rejected(tmp_path: Path) -> None:
    body = """
global_seed: 1
attack_budget: 0.1
budget_scope: [test]
enabled_families: []
min_window_length: 2
max_window_length: 6
max_attempts: 10
family_parameters: {}
"""
    with pytest.raises(ConfigError, match="at least 1"):
        load_scheduling_config(_write(tmp_path, body))


def test_duplicate_family_is_rejected(tmp_path: Path) -> None:
    body = """
global_seed: 1
attack_budget: 0.1
budget_scope: [test]
enabled_families: [spike, spike]
min_window_length: 2
max_window_length: 6
max_attempts: 10
family_parameters:
  spike: {weight: 1.0}
"""
    with pytest.raises(ConfigError, match="duplicate"):
        load_scheduling_config(_write(tmp_path, body))


def test_allow_overlap_true_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not supported"):
        load_scheduling_config(_write(tmp_path, _VALID + "allow_overlap: true\n"))


def test_weight_for_disabled_family_is_rejected(tmp_path: Path) -> None:
    body = _VALID + "  drift: {weight: 0.5}\n"
    with pytest.raises(ConfigError, match="disabled families"):
        load_scheduling_config(_write(tmp_path, body))


def test_missing_family_parameters_is_rejected(tmp_path: Path) -> None:
    body = """
global_seed: 1
attack_budget: 0.1
budget_scope: [test]
enabled_families: [spike, bias]
min_window_length: 2
max_window_length: 6
max_attempts: 10
family_parameters:
  spike: {weight: 1.0}
"""
    with pytest.raises(ConfigError, match="missing family_parameters"):
        load_scheduling_config(_write(tmp_path, body))


def test_inverted_global_window_bounds_rejected() -> None:
    with pytest.raises(ValueError, match="min_window_length"):
        SchedulingConfig(
            global_seed=1,
            attack_budget=0.1,
            budget_scope=("test",),
            enabled_families=(AttackFamily.SPIKE,),
            family_parameters={AttackFamily.SPIKE: {"weight": 1.0}},
            min_window_length=10,
            max_window_length=5,
            max_attempts=10,
        )


def test_inverted_family_window_bounds_rejected() -> None:
    with pytest.raises(ValueError, match="min_window must not exceed max_window"):
        SchedulingConfig(
            global_seed=1,
            attack_budget=0.1,
            budget_scope=("test",),
            enabled_families=(AttackFamily.SPIKE,),
            family_parameters={
                AttackFamily.SPIKE: {"weight": 1.0, "min_window": 6, "max_window": 3}
            },
            min_window_length=1,
            max_window_length=10,
            max_attempts=10,
        )


def test_effective_bounds_from_override_are_checked() -> None:
    # Family min_window override (10) exceeds the inherited global max_window (5).
    with pytest.raises(ValueError, match="inverted"):
        SchedulingConfig(
            global_seed=1,
            attack_budget=0.1,
            budget_scope=("test",),
            enabled_families=(AttackFamily.SPIKE,),
            family_parameters={AttackFamily.SPIKE: {"weight": 1.0, "min_window": 10}},
            min_window_length=1,
            max_window_length=5,
            max_attempts=10,
        )


def test_duplicate_scope_rejected() -> None:
    with pytest.raises(ValueError, match="budget_scope"):
        SchedulingConfig(
            global_seed=1,
            attack_budget=0.1,
            budget_scope=("test", "test"),
            enabled_families=(AttackFamily.SPIKE,),
            family_parameters={AttackFamily.SPIKE: {"weight": 1.0}},
            min_window_length=1,
            max_window_length=10,
            max_attempts=10,
        )


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_scheduling_config(Path("does-not-exist.yaml"))
