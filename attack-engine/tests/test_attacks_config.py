from __future__ import annotations

from pathlib import Path

import pytest

from attack_engine.attacks.config import (
    AttackConfigError,
    AttackExecutionConfig,
    load_attack_config,
)
from attack_engine.scheduling.models import AttackFamily

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "attacks.yaml"


def test_project_config_loads() -> None:
    config = load_attack_config(PROJECT_CONFIG)
    assert config.global_seed == 14
    assert config.quantum == 0.001
    assert config.parameters_for(AttackFamily.SPIKE).magnitude == 0.30
    assert config.parameters_for(AttackFamily.BIAS).magnitude == 0.05
    assert config.parameters_for(AttackFamily.DRIFT).final_magnitude == 0.10
    assert config.parameters_for(AttackFamily.REPLAY).min_age_hours == 24


def test_stealth_is_executable_and_configured() -> None:
    config = load_attack_config(PROJECT_CONFIG)
    assert config.parameters_for(AttackFamily.STEALTH) is config.stealth
    assert config.stealth.calibration_path.is_absolute()


def test_missing_parameters_is_reported() -> None:
    config = AttackExecutionConfig(global_seed=1)
    with pytest.raises(AttackConfigError, match="no parameters configured"):
        config.parameters_for(AttackFamily.SPIKE)
    assert not config.has_parameters(AttackFamily.SPIKE)


@pytest.mark.parametrize(
    "body",
    [
        "global_seed: 1\nspike: {magnitude: -0.1}",  # magnitude must be > 0
        "global_seed: 1\nspike: {magnitude: 0.1, min_points: 3, max_points: 1}",  # inverted
        "global_seed: -1",  # seed must be >= 0
        "global_seed: 1\ntarget_low: 0.8\ntarget_high: 0.7",  # inverted bounds
        "global_seed: 1\nquantum: 0",  # quantum must be > 0
        "global_seed: 1\nquantum: .inf",  # grid must be finite
        "global_seed: 1\nquantum: .nan",
        "global_seed: 1\nspike: {magnitude: .inf}",
        "global_seed: 1\ntarget_low: .nan",
        "global_seed: 1\ndrift: {final_magnitude: 0.1, max_slope_per_hour: .inf}",
        "global_seed: 1\nunknown_field: 2",  # extra field
        "global_seed: 1\ndrift: {final_magnitude: 0.1, max_slope_per_hour: -1}",
    ],
)
def test_invalid_configs_rejected(tmp_path: Path, body: str) -> None:
    path = tmp_path / "attacks.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(AttackConfigError):
        load_attack_config(path)


def test_missing_file() -> None:
    with pytest.raises(AttackConfigError, match="not found"):
        load_attack_config(Path("nope.yaml"))


def test_non_mapping_rejected(tmp_path: Path) -> None:
    path = tmp_path / "attacks.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(AttackConfigError, match="mapping"):
        load_attack_config(path)
