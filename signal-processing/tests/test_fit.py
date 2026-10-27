"""Train-only fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_processing.fit import fit_params
from signal_processing.io import epoch_hours
from signal_processing.params import FrozenParams
from signal_processing.partition import DEFAULT_RULES, train_mask


def _fit(frame: pd.DataFrame) -> FrozenParams:
    return fit_params(frame, "0" * 64, DEFAULT_RULES, "built-in defaults")


def test_params_ignore_validation_and_test_rows(frame: pd.DataFrame, params: FrozenParams) -> None:
    rng = np.random.default_rng(7)
    perturbed = frame.copy()
    later = ~train_mask(epoch_hours(frame["time"]), DEFAULT_RULES)
    assert later.any()
    for column in ("SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "P_CALC"):
        perturbed.loc[later, column] = rng.uniform(0, 0.6, later.sum()).round(3)
    perturbed.loc[np.flatnonzero(later)[::7], "SOIL_MOISTURE_5"] = np.nan
    refit = _fit(perturbed)
    assert refit.model_dump(exclude={"input_rows"}) == params.model_dump(exclude={"input_rows"})


def test_drydown_recovers_reasonable_parameters(params: FrozenParams) -> None:
    for station in params.stations.values():
        dd = station.depths["5"].drydown
        assert 50 < dd.tau_hours < 2000
        assert dd.k_rain_per_mm >= 0
        assert station.depths["5"].innov_scale >= params.constants.scale_floor
        assert station.depths["5"].cusum_drift >= params.constants.cusum_drift_min
        assert np.isfinite(station.depths["5"].cusum_h)


def test_partition_column_must_agree(frame: pd.DataFrame) -> None:
    import pytest

    wrong = frame.assign(partition="validation")
    with pytest.raises(ValueError, match="disagrees"):
        _fit(wrong)


def test_no_replay_matches_in_clean_train(params: FrozenParams) -> None:
    assert all(sp.train_replay_matches == 0 for sp in params.stations.values())
