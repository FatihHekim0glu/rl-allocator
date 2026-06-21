"""Unit tests for the serve entrypoint, the CLI, the plot builders, and the stubs."""

from __future__ import annotations

import numpy as np
import pytest

from rlallocator._exceptions import ArtifactError, ValidationError
from rlallocator.agents.onnx_policy import OnnxPolicy, default_artifact_path
from rlallocator.agents.ppo import PpoAgent, PpoConfig
from rlallocator.plots import (
    equity_curve_figure,
    seed_lottery_figure,
    weights_area_figure,
)
from rlallocator.serve import RlAllocatorRun, run_allocation
from rlallocator.train import n_effective_trials, train_pipeline


@pytest.mark.unit
def test_run_allocation_returns_run_with_baselines() -> None:
    """run_allocation returns a JSON-safe run with the three live baseline Sharpes."""
    run = run_allocation(n_assets=5, n_seeds=4, cost_bps=10.0, lookback=16, seed=7)
    assert isinstance(run, RlAllocatorRun)
    s = run.summary
    assert s.data_source == "synthetic"
    assert s.best_baseline in {"equal_weight", "markowitz", "risk_parity"}
    # No committed artifact: the honest-null placeholder holds.
    assert s.rl_beats_baselines is False
    payload = run.to_dict()
    assert set(payload) == {"summary", "equity_figure", "weights_figure"}


@pytest.mark.unit
def test_run_allocation_validates_request() -> None:
    """run_allocation enforces the request caps."""
    with pytest.raises(ValidationError):
        run_allocation(n_assets=1)
    with pytest.raises(ValidationError):
        run_allocation(n_seeds=99)
    with pytest.raises(ValidationError):
        run_allocation(cost_bps=-1.0)


@pytest.mark.unit
def test_equity_curve_figure_shape() -> None:
    """The equity figure serializes to a {data, layout} mapping with all traces."""
    rl = np.cumprod(1.0 + np.full(20, 0.001))
    baselines = {
        "equal_weight": (1.0 + np.full(20, 0.0005)).cumprod().tolist(),
        "markowitz": (1.0 + np.full(20, 0.0004)).cumprod().tolist(),
        "risk_parity": (1.0 + np.full(20, 0.0006)).cumprod().tolist(),
    }
    fig = equity_curve_figure(rl_median_equity=rl, baseline_equities=baselines)
    assert "data" in fig and "layout" in fig
    assert len(fig["data"]) == 4  # RL + three baselines


@pytest.mark.unit
def test_weights_area_figure_shape() -> None:
    """The weights area figure has one stacked trace per asset."""
    rng = np.random.default_rng(0)
    raw = rng.random((30, 4))
    weight_path = raw / raw.sum(axis=1, keepdims=True)
    fig = weights_area_figure(weight_path)
    assert len(fig["data"]) == 4


@pytest.mark.unit
def test_seed_lottery_figure_shape() -> None:
    """The seed-lottery figure serializes to a {data, layout} mapping."""
    fig = seed_lottery_figure(np.array([-0.1, 0.2, 0.0, 0.3]), best_baseline_sharpe=0.4)
    assert "data" in fig and "layout" in fig


@pytest.mark.unit
def test_n_effective_trials_counts_seeds_times_hp() -> None:
    """n_effective_trials counts the full seed x HP grid."""
    assert n_effective_trials(5) == 5
    with pytest.raises(ValidationError):
        n_effective_trials(0)


@pytest.mark.unit
def test_train_pipeline_is_scaffold_stub() -> None:
    """The offline train pipeline is a scaffold stub (NotImplementedError)."""
    with pytest.raises(NotImplementedError):
        train_pipeline(n_seeds=2)


@pytest.mark.unit
def test_ppo_agent_stub_raises() -> None:
    """The PPO agent training / export are scaffold stubs (NotImplementedError)."""
    agent = PpoAgent(PpoConfig(obs_dim=10, n_assets=3))
    assert not agent.is_trained
    with pytest.raises(NotImplementedError):
        agent.train(env=object())


@pytest.mark.unit
def test_onnx_policy_missing_artifact_raises_artifact_error() -> None:
    """OnnxPolicy.load raises ArtifactError when the committed artifact is absent."""
    assert default_artifact_path().name == "policy.onnx"
    if not default_artifact_path().is_file():
        with pytest.raises(ArtifactError):
            OnnxPolicy().load()
