"""Unit tests for env rollout/validation, parity helpers, loaders, and onnx-policy stubs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlallocator._exceptions import ArtifactError, InsufficientDataError, ValidationError
from rlallocator.agents.onnx_policy import OnnxPolicy
from rlallocator.data.loaders import load_multi_asset_panel
from rlallocator.env.parity import check_parity, leaky_backtest
from rlallocator.env.portfolio_env import PortfolioEnv, PortfolioEnvConfig


@pytest.mark.unit
def test_env_rollout_matches_step_loop(synthetic_panel: pd.DataFrame) -> None:
    """The env rollout reproduces a manual reset/step loop's per-bar net rewards."""
    arr = synthetic_panel.to_numpy()[:60]
    cfg = PortfolioEnvConfig(lookback=1, cost_bps=10.0)
    rng = np.random.default_rng(0)
    actions = rng.normal(size=arr.shape)
    rollout_net = PortfolioEnv(arr, cfg).rollout(actions)

    env = PortfolioEnv(arr, cfg)
    env.reset()
    manual = []
    for t in range(arr.shape[0] - 1):
        manual.append(env.step(actions[t]).reward)
    assert np.allclose(rollout_net, manual, atol=1e-12)


@pytest.mark.unit
def test_env_rollout_shape_guard(synthetic_panel: pd.DataFrame) -> None:
    """rollout rejects an action path that does not match the panel shape."""
    arr = synthetic_panel.to_numpy()[:20]
    env = PortfolioEnv(arr, PortfolioEnvConfig(lookback=2))
    with pytest.raises(ValidationError):
        env.rollout(np.zeros((5, 2)))


@pytest.mark.unit
def test_env_step_rejects_wrong_action_width(synthetic_panel: pd.DataFrame) -> None:
    """step rejects an action vector whose length is not n_assets."""
    arr = synthetic_panel.to_numpy()
    env = PortfolioEnv(arr, PortfolioEnvConfig(lookback=4))
    env.reset()
    with pytest.raises(ValidationError):
        env.step(np.array([1.0, 0.0]))  # too few assets


@pytest.mark.unit
def test_env_too_short_panel_raises() -> None:
    """A panel shorter than lookback + 1 raises on reset."""
    env = PortfolioEnv(np.zeros((3, 4)), PortfolioEnvConfig(lookback=10))
    with pytest.raises(InsufficientDataError):
        env.reset()


@pytest.mark.unit
def test_env_long_short_budget() -> None:
    """In the long/short regime the resolved weights sum-of-abs is one (unit budget)."""
    arr = np.random.default_rng(0).normal(0.0, 0.01, size=(20, 4))
    env = PortfolioEnv(arr, PortfolioEnvConfig(lookback=2, long_only=False))
    path = env.resolved_weight_path(np.random.default_rng(1).normal(size=arr.shape))
    assert np.allclose(np.abs(path).sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.unit
def test_check_parity_rejects_bad_tol(synthetic_panel: pd.DataFrame) -> None:
    """check_parity rejects a negative tolerance."""
    arr = synthetic_panel.to_numpy()[:30]
    with pytest.raises(ValidationError):
        check_parity(arr, np.zeros(arr.shape), tol=-1.0)


@pytest.mark.unit
def test_leaky_backtest_differs_from_honest(synthetic_panel: pd.DataFrame) -> None:
    """The leaky control's net returns differ from the honest vectorized backtest."""
    from rlallocator.env.backtester import vectorized_backtest

    arr = synthetic_panel.to_numpy()[:40]
    rng = np.random.default_rng(0)
    weights = rng.normal(size=arr.shape)
    env = PortfolioEnv(arr, PortfolioEnvConfig(lookback=1, cost_bps=10.0))
    projected = env.resolved_weight_path(weights)
    honest = vectorized_backtest(arr, projected, cost_bps=10.0).net_returns
    leaky = leaky_backtest(arr, weights, cost_bps=10.0)
    assert not np.allclose(honest, leaky)


@pytest.mark.unit
def test_loaders_reject_bad_tickers() -> None:
    """load_multi_asset_panel rejects too-few / duplicate tickers and a bad date range."""
    from datetime import date

    with pytest.raises(ValidationError):
        load_multi_asset_panel(["SPY"], start=date(2020, 1, 1), end=date(2021, 1, 1))
    with pytest.raises(ValidationError):
        load_multi_asset_panel(["SPY", "SPY"], start=date(2020, 1, 1), end=date(2021, 1, 1))
    with pytest.raises(ValidationError):
        load_multi_asset_panel(["SPY", "TLT"], start=date(2021, 1, 1), end=date(2020, 1, 1))


@pytest.mark.unit
def test_loaders_eodhd_falls_back_to_synthetic() -> None:
    """The EODHD path (no wired reader) falls back to the deterministic synthetic panel."""
    from datetime import date

    _, returns, source = load_multi_asset_panel(
        ["A", "B", "C"],
        start=date(2018, 1, 1),
        end=date(2019, 1, 1),
        data_source_pref="eodhd",
    )
    assert source == "synthetic"
    assert returns.shape[1] == 3


@pytest.mark.unit
def test_onnx_policy_predict_stub_raises() -> None:
    """OnnxPolicy.predict_scores / predict_weights are scaffold stubs that raise."""
    policy = OnnxPolicy()
    with pytest.raises((NotImplementedError, ArtifactError)):
        policy.predict_scores(np.zeros((2, 10)))
