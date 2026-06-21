"""Unit tests for the multi-asset portfolio env + the vectorized backtester."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rlallocator._exceptions import InsufficientDataError, ValidationError
from rlallocator._validation import is_simplex
from rlallocator.env.backtester import equity_curve, vectorized_backtest
from rlallocator.env.portfolio_env import (
    PortfolioEnv,
    PortfolioEnvConfig,
    weights_are_simplex,
)


@pytest.mark.unit
def test_env_reset_and_step(synthetic_panel: pd.DataFrame) -> None:
    """reset returns a well-formed observation; step returns a causal reward + simplex."""
    env = PortfolioEnv(synthetic_panel.to_numpy(), PortfolioEnvConfig(lookback=8, cost_bps=10.0))
    obs, _info = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    result = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert math.isfinite(result.reward)
    assert is_simplex(result.info["weights"])


@pytest.mark.unit
def test_env_obs_dim_formula(synthetic_panel: pd.DataFrame) -> None:
    """obs_dim == lookback * n_assets + n_assets."""
    env = PortfolioEnv(synthetic_panel.to_numpy(), PortfolioEnvConfig(lookback=8))
    assert env.obs_dim == 8 * env.n_assets + env.n_assets


@pytest.mark.unit
def test_env_step_after_done_raises(synthetic_panel: pd.DataFrame) -> None:
    """Stepping a finished episode raises."""
    arr = synthetic_panel.to_numpy()[:10]
    env = PortfolioEnv(arr, PortfolioEnvConfig(lookback=2), episode_len=3)
    env.reset()
    flat = np.ones(env.n_assets)
    for _ in range(3):
        env.step(flat)
    with pytest.raises(ValidationError):
        env.step(flat)


@pytest.mark.unit
def test_resolved_weight_path_is_simplex_every_bar(synthetic_panel: pd.DataFrame) -> None:
    """Every row of the resolved weight path is a valid simplex."""
    arr = synthetic_panel.to_numpy()
    env = PortfolioEnv(arr, PortfolioEnvConfig(lookback=4))
    rng = np.random.default_rng(0)
    actions = rng.normal(size=arr.shape)
    path = env.resolved_weight_path(actions)
    assert weights_are_simplex(path)


@pytest.mark.unit
def test_vectorized_backtest_causal_reward() -> None:
    """The net return at t is w_t · r_{t+1} - cost·||Δw||_1 (strictly causal)."""
    returns = np.array([[0.0, 0.0], [0.1, -0.1], [0.2, 0.0]], dtype="float64")
    weights = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype="float64")
    result = vectorized_backtest(returns, weights, cost_bps=0.0)
    # gross_0 = w_0 · r_1 = 1*0.1 = 0.1; gross_1 = w_1 · r_2 = 1*0.2 = 0.2
    assert np.allclose(result.gross_returns, [0.1, 0.2])
    assert result.n_bars == 2


@pytest.mark.unit
def test_vectorized_backtest_turnover_cost() -> None:
    """Turnover cost is charged on the L1 weight change against the all-cash open."""
    returns = np.array([[0.0, 0.0], [0.0, 0.0]], dtype="float64")
    weights = np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float64")
    result = vectorized_backtest(returns, weights, cost_bps=100.0)  # 1% per unit turnover
    # bar0 turnover = |0.5| + |0.5| = 1.0 against flat; cost = 1.0 * 100/1e4 = 0.01
    assert math.isclose(result.costs[0], 0.01)
    assert math.isclose(result.net_returns[0], -0.01)


@pytest.mark.unit
def test_backtester_rejects_shape_mismatch() -> None:
    """Mismatched returns/weights shapes are rejected."""
    with pytest.raises(ValidationError):
        vectorized_backtest(np.zeros((5, 3)), np.zeros((5, 2)))


@pytest.mark.unit
def test_backtester_needs_two_bars() -> None:
    """A single-bar panel cannot score a causal step."""
    with pytest.raises(InsufficientDataError):
        vectorized_backtest(np.zeros((1, 3)), np.zeros((1, 3)))


@pytest.mark.unit
def test_equity_curve_compounds() -> None:
    """The equity curve is the cumulative product of (1 + net)."""
    net = np.array([0.1, -0.05])
    assert np.allclose(equity_curve(net), np.cumprod(1.0 + net))
