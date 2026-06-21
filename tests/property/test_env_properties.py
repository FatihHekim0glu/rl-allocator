"""Property tests (Hypothesis): causal reward, simplex validity, parity, cost monotonicity."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rlallocator._validation import is_simplex, project_to_simplex
from rlallocator.env.backtester import vectorized_backtest
from rlallocator.env.parity import check_parity
from rlallocator.env.portfolio_env import PortfolioEnv, PortfolioEnvConfig

_PROP_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.mark.property
@given(scores=st.lists(st.floats(-10.0, 10.0), min_size=2, max_size=12))
@_PROP_SETTINGS
def test_projection_is_always_a_valid_simplex(scores: list[float]) -> None:
    """project_to_simplex ALWAYS produces a valid long-only simplex."""
    w = project_to_simplex(np.asarray(scores, dtype="float64"))
    assert is_simplex(w)


@pytest.mark.property
@given(
    seed=st.integers(0, 2**16),
    n_bars=st.integers(20, 60),
    n_assets=st.integers(2, 6),
)
@_PROP_SETTINGS
def test_backtester_matches_env_for_random_weight_paths(
    seed: int, n_bars: int, n_assets: int
) -> None:
    """For random weight paths the vectorized backtester == the env rollout to 1e-10."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=(n_bars, n_assets))
    weights = rng.normal(size=(n_bars, n_assets))
    report = check_parity(returns, weights, cost_bps=10.0)
    assert report.passed
    assert report.max_abs_diff <= 1e-10


@pytest.mark.property
@given(seed=st.integers(0, 2**16))
@_PROP_SETTINGS
def test_causal_reward_future_perturbation_invariance(seed: int) -> None:
    """Perturbing a FUTURE bar does not change the reward earned at the current bar."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=(30, 4))
    env = PortfolioEnv(returns.copy(), PortfolioEnvConfig(lookback=3, cost_bps=10.0))
    env.reset()
    action = rng.normal(size=4)
    r_original = env.step(action).reward

    # Perturb a strictly-future bar (the last one) and re-run the FIRST step.
    perturbed = returns.copy()
    perturbed[-1] += 5.0
    env2 = PortfolioEnv(perturbed, PortfolioEnvConfig(lookback=3, cost_bps=10.0))
    env2.reset()
    r_perturbed = env2.step(action).reward
    assert abs(r_original - r_perturbed) <= 1e-12


@pytest.mark.property
@given(seed=st.integers(0, 2**16))
@_PROP_SETTINGS
def test_cost_monotonicity(seed: int) -> None:
    """Higher turnover cost yields a non-greater total net return (cost monotonicity)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=(40, 4))
    # A churny weight path that actually trades each bar.
    raw = rng.normal(size=(40, 4))
    weights = np.asarray([project_to_simplex(row) for row in raw], dtype="float64")
    low = vectorized_backtest(returns, weights, cost_bps=1.0).net_returns.sum()
    high = vectorized_backtest(returns, weights, cost_bps=50.0).net_returns.sum()
    assert high <= low + 1e-12
