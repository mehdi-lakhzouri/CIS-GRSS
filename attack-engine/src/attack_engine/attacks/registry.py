"""Dispatch table from attack family to its transformation callable."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from attack_engine.attacks.bias import apply_bias
from attack_engine.attacks.drift import apply_drift
from attack_engine.attacks.models import AttackRequest, AttackResult, AttackSettings
from attack_engine.attacks.replay import apply_replay
from attack_engine.attacks.spike import apply_spike
from attack_engine.attacks.stealth import apply_stealth
from attack_engine.scheduling.models import AttackFamily

# Each transform shares the signature (request, params, rng, settings) -> AttackResult;
# ``params`` is the family's own validated parameter model.
Transform = Callable[[AttackRequest, object, np.random.Generator, AttackSettings], AttackResult]

ATTACK_TRANSFORMS: dict[AttackFamily, Transform] = {
    AttackFamily.SPIKE: apply_spike,
    AttackFamily.BIAS: apply_bias,
    AttackFamily.DRIFT: apply_drift,
    AttackFamily.REPLAY: apply_replay,
    AttackFamily.STEALTH: apply_stealth,
}


def transform_for(family: AttackFamily) -> Transform:
    try:
        return ATTACK_TRANSFORMS[family]
    except KeyError:
        raise KeyError(f"no transformation registered for attack family {family.value!r}") from None
