"""Regression: the parity oracle CATCHES the deliberately-leaky negative control.

This proves the parity oracle is not vacuous: a backtester that scores the weights at
``t`` against the CONTEMPORANEOUS ``r_t`` (look-ahead) instead of the next-bar
``r_{t+1}`` MUST be caught by the oracle (a :class:`ParityError`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlallocator._exceptions import ParityError
from rlallocator.env.parity import (
    assert_parity,
    assert_parity_against,
    leaky_backtest,
)


@pytest.mark.regression
def test_leaky_backtest_is_caught_by_the_oracle(synthetic_panel: pd.DataFrame) -> None:
    """The leaky (look-ahead) backtester disagrees with the env rollout — the oracle catches it."""
    arr = synthetic_panel.to_numpy()[:120]
    rng = np.random.default_rng(0)
    # A churny weight path so the leaked-vs-honest difference is non-degenerate.
    weights = rng.normal(size=arr.shape)
    leaky = leaky_backtest(arr, weights, cost_bps=10.0)
    with pytest.raises(ParityError):
        assert_parity_against(leaky, arr, weights, cost_bps=10.0)


@pytest.mark.regression
def test_honest_path_passes_the_same_oracle(synthetic_panel: pd.DataFrame) -> None:
    """The honest path passes the same oracle the leaky control fails (no false alarm)."""
    arr = synthetic_panel.to_numpy()[:120]
    rng = np.random.default_rng(0)
    weights = rng.normal(size=arr.shape)
    # assert_parity must NOT raise on the honest vectorized backtester.
    net = assert_parity(arr, weights, cost_bps=10.0)
    assert net.shape == (arr.shape[0] - 1,)
