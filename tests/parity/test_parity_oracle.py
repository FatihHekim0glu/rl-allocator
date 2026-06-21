"""Parity: the vectorized backtester == the step-by-step env rollout to 1e-10."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlallocator.env.parity import PARITY_TOL, assert_parity, check_parity


@pytest.mark.parity
def test_vectorized_equals_stepwise_1e10(synthetic_panel: pd.DataFrame) -> None:
    """For a random weight path the two paths agree to 1e-10 (the look-ahead guard)."""
    arr = synthetic_panel.to_numpy()[:200]
    rng = np.random.default_rng(0)
    weights = rng.normal(size=arr.shape)  # raw scores, projected identically by both paths
    report = check_parity(arr, weights, cost_bps=10.0)
    assert report.passed
    assert report.max_abs_diff <= PARITY_TOL


@pytest.mark.parity
def test_assert_parity_returns_agreed_curve(synthetic_panel: pd.DataFrame) -> None:
    """assert_parity returns the agreed per-bar net-reward series on success."""
    arr = synthetic_panel.to_numpy()[:120]
    rng = np.random.default_rng(1)
    weights = rng.normal(size=arr.shape)
    net = assert_parity(arr, weights, cost_bps=5.0)
    assert net.shape == (arr.shape[0] - 1,)
    assert np.isfinite(net).all()


@pytest.mark.parity
@pytest.mark.parametrize("cost_bps", [0.0, 5.0, 20.0])
def test_parity_across_costs(synthetic_panel: pd.DataFrame, cost_bps: float) -> None:
    """Parity holds across cost levels (costs are charged identically in both paths)."""
    arr = synthetic_panel.to_numpy()[:150]
    rng = np.random.default_rng(2)
    weights = rng.normal(size=arr.shape)
    assert check_parity(arr, weights, cost_bps=cost_bps).passed
