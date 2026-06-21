"""Parity: the exported policy ONNX graph reproduces the torch policy MLP to 1e-4.

The SB3 PPO allocator is trained offline (the ``[train]`` extra: torch + sb3 +
gymnasium), its small policy MLP (pi feature extractor + per-asset score head) is
EXTRACTED into a standalone plain-torch module, exported to ONNX with the LEGACY
exporter (``dynamo=False`` — the dynamo path needs onnxscript, which is NOT a
dependency), and served torch-free through onnxruntime via
:class:`rlallocator.agents.onnx_policy.OnnxPolicy`. This pins the load-bearing contract
that the SERVED ONNX policy is numerically faithful to the torch policy (the 1e-4 gate
the brief mandates) and that the export's batch axis is DYNAMIC so any per-bar
observation batch binds at inference; the served scores project to valid simplex
weights every bar.

These are marked ``slow`` (they build / train a torch model) and are SKIPPED where
torch is absent (the lean CI ``[serve]`` image runs the torch-free serve tests only).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rlallocator._validation import is_simplex
from rlallocator.agents.onnx_policy import OnnxPolicy
from rlallocator.agents.ppo import PpoAgent, PpoConfig, _build_policy_mlp

# torch gates the whole module (the [train] extra); when it is absent the module is
# skipped at collection and onnxruntime is never touched. onnxruntime is checked with
# ``find_spec`` (no import) so collecting this module does not load the engine into the
# parent process; the genuine session is created inside OnnxPolicy in the test bodies.
torch = pytest.importorskip("torch")
if importlib.util.find_spec("onnxruntime") is None:  # pragma: no cover - lean image guard
    pytest.skip("onnxruntime ([serve] extra) not installed", allow_module_level=True)

#: The torch-vs-ONNX policy parity gate the brief mandates.
_PARITY_RTOL: float = 1e-4

#: A small obs_dim keeps the slow export fast while still exercising the MLP head.
#: obs_dim = lookback * n_assets + n_assets = 2 * 4 + 4 = 12.
_SMALL_CFG = PpoConfig(obs_dim=12, n_assets=4, hidden_dim=8)


def _probe_batch(obs_dim: int) -> np.ndarray:
    """A deterministic, varied observation batch for the parity comparison."""
    rng = np.random.default_rng(20260621)
    return rng.standard_normal((16, obs_dim)).astype("float64")


@pytest.mark.parity
@pytest.mark.slow
def test_export_onnx_torch_parity_via_extracted_mlp(tmp_path: Path) -> None:
    """An untrained-but-randomly-weighted policy MLP exports + serves to 1e-4 parity.

    Exercises the export + parity gate without SB3: build the standalone policy MLP
    directly, install it on the agent, export, and assert the ONNX serve scores match
    the torch forward pass to the 1e-4 gate. Covers the export path where only torch
    (not the full ``[train]`` stack) is available.
    """
    agent = PpoAgent(_SMALL_CFG)
    net = _build_policy_mlp(_SMALL_CFG)
    net.eval()
    agent._policy_net = net  # inject the policy MLP directly (no SB3 needed).

    out = agent.export_onnx(tmp_path / "policy.onnx", rtol=_PARITY_RTOL)
    assert out.is_file()

    probe = _probe_batch(_SMALL_CFG.obs_dim)
    with torch.no_grad():
        torch_scores = np.asarray(net(torch.from_numpy(probe.astype("float32"))).numpy())
    onnx_scores = OnnxPolicy(out).predict_scores(probe)

    assert onnx_scores.shape == (16, _SMALL_CFG.n_assets)
    np.testing.assert_allclose(onnx_scores, torch_scores, rtol=_PARITY_RTOL, atol=_PARITY_RTOL)

    # policy_scores (the torch reference) and the ONNX serve agree to the same gate.
    ref = agent.policy_scores(probe)
    np.testing.assert_allclose(ref, onnx_scores, rtol=_PARITY_RTOL, atol=_PARITY_RTOL)

    # The served weight path is a valid long-only simplex EVERY bar.
    weights = OnnxPolicy(out).predict_weights(probe)
    assert weights.shape == (16, _SMALL_CFG.n_assets)
    assert all(is_simplex(row, long_only=True) for row in weights)


@pytest.mark.parity
@pytest.mark.slow
def test_export_onnx_dynamic_batch_axis(tmp_path: Path) -> None:
    """The exported graph binds an ARBITRARY batch (the dynamic batch axis contract)."""
    agent = PpoAgent(_SMALL_CFG)
    net = _build_policy_mlp(_SMALL_CFG)
    net.eval()
    agent._policy_net = net
    out = agent.export_onnx(tmp_path / "policy.onnx", rtol=_PARITY_RTOL)

    policy = OnnxPolicy(out)
    for batch in (1, 3, 9):
        scores = policy.predict_scores(np.zeros((batch, _SMALL_CFG.obs_dim)))
        assert scores.shape == (batch, _SMALL_CFG.n_assets)


@pytest.mark.parity
@pytest.mark.slow
def test_sb3_train_extract_export_parity(tmp_path: Path) -> None:
    """The full SB3 PPO -> policy-MLP extract -> ONNX export path holds 1e-4 parity.

    Trains a tiny SB3 PPO for a handful of timesteps on the gym-wrapped portfolio env,
    extracts the policy MLP, exports it to ONNX, and asserts the served ONNX scores
    match the extracted torch policy to 1e-4 and project to valid simplex weights.
    Skipped when sb3 / gymnasium are absent.
    """
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")

    from rlallocator.env.portfolio_env import PortfolioEnv, PortfolioEnvConfig

    rng = np.random.default_rng(7)
    returns = (0.0005 + 0.01 * rng.standard_normal((400, 4))).astype("float64")
    env = PortfolioEnv(returns, PortfolioEnvConfig(lookback=2)).as_gym_env()

    # A tiny budget keeps the slow test bounded; obs_dim = lookback*n_assets + n_assets.
    cfg = PpoConfig(obs_dim=12, n_assets=4, hidden_dim=8, n_steps=64, total_timesteps=128)
    agent = PpoAgent(cfg).train(env, seed=7)
    assert agent.is_trained is True

    out = agent.export_onnx(tmp_path / "policy.onnx", rtol=_PARITY_RTOL)
    assert out.is_file()

    probe = _probe_batch(cfg.obs_dim)
    torch_scores = agent.policy_scores(probe)
    onnx_scores = OnnxPolicy(out).predict_scores(probe)
    np.testing.assert_allclose(onnx_scores, torch_scores, rtol=_PARITY_RTOL, atol=_PARITY_RTOL)

    # Served weights are valid simplices every bar.
    weights = OnnxPolicy(out).predict_weights(probe)
    assert all(is_simplex(row, long_only=True) for row in weights)
