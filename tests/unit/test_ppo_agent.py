"""Torch-free unit tests for the PPO agent surface (:mod:`rlallocator.agents.ppo`).

The PPO trainer is the OFFLINE ``[train]``-extra agent (torch + stable-baselines3 +
gymnasium, all LAZY). These tests pin the import-pure, torch-FREE surface that holds
WITHOUT the heavy extras installed:

- :class:`PpoConfig` is a frozen, JSON-serializable dataclass whose default
  ``obs_dim`` matches the env observation signature
  (``lookback * n_assets + n_assets``);
- constructing a :class:`PpoAgent` imports no torch / sb3 and yields an untrained agent;
- calling :meth:`policy_scores` / :meth:`export_onnx` before training raises a clean
  :class:`ValidationError` (NOT an obscure attribute error);
- :meth:`export_onnx` validates ``rtol`` before importing torch;
- :meth:`train` rejects a negative seed before any heavy import, and (when the
  ``[train]`` extra is absent) raises ``NotImplementedError`` rather than a raw
  ``ImportError`` — the offline path is simply unavailable on the lean serve image.

The actual SB3 train -> policy-MLP extract -> ONNX export is exercised in the ``slow``
parity suite (skipped where the ``[train]`` extra is absent).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from rlallocator._exceptions import ValidationError
from rlallocator.agents.ppo import ACTION_OUTPUT, OBSERVATION_INPUT, ONNX_OPSET, PpoAgent, PpoConfig
from rlallocator.env.portfolio_env import PortfolioEnvConfig

#: Whether the [train] extra (torch + sb3) is present in this environment.
_HAS_TRAIN: bool = bool(importlib.util.find_spec("torch"))


@pytest.mark.unit
def test_ppo_config_defaults_and_to_dict_are_json_safe() -> None:
    """``PpoConfig`` defaults round-trip through a plain JSON-safe dict."""
    cfg = PpoConfig()
    payload = cfg.to_dict()
    assert payload["obs_dim"] == 390
    assert payload["n_assets"] == 6
    assert payload["hidden_dim"] == 64
    # All values are plain Python scalars (JSON-serializable).
    assert all(isinstance(v, (int, float)) for v in payload.values())


@pytest.mark.unit
def test_ppo_config_obs_dim_matches_default_env_observation() -> None:
    """The policy ``obs_dim`` matches the default env observation width.

    The env observation is the FLATTENED look-back window of per-asset returns
    (``lookback * n_assets``) PLUS the current weight vector (``n_assets``).
    """
    env_cfg = PortfolioEnvConfig()
    cfg = PpoConfig()
    assert cfg.obs_dim == env_cfg.lookback * cfg.n_assets + cfg.n_assets


@pytest.mark.unit
def test_ppo_config_is_frozen() -> None:
    """``PpoConfig`` is immutable (a frozen dataclass)."""
    cfg = PpoConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.obs_dim = 7  # type: ignore[misc]


@pytest.mark.unit
def test_onnx_signature_constants_are_stable() -> None:
    """The ONNX input/output names + opset pin the exported-graph signature."""
    assert OBSERVATION_INPUT == "observation"
    assert ACTION_OUTPUT == "asset_scores"
    assert ONNX_OPSET == 17


@pytest.mark.unit
def test_ppo_agent_construction_is_import_pure() -> None:
    """Constructing a ``PpoAgent`` imports no NEW torch/sb3 and yields an untrained agent."""
    # White-box: snapshot whether the heavy modules are already loaded (a prior slow
    # parity test in the same process may have imported torch) so we assert that
    # CONSTRUCTING the agent imports no NEW heavy module — order-independent.
    torch_before = "torch" in sys.modules
    sb3_before = "stable_baselines3" in sys.modules

    agent = PpoAgent()
    assert agent.is_trained is False
    assert agent.config == PpoConfig()
    # Constructing the agent loads no heavy module that was not already present.
    assert ("torch" in sys.modules) == torch_before
    assert ("stable_baselines3" in sys.modules) == sb3_before


@pytest.mark.unit
def test_custom_config_is_recorded() -> None:
    """A custom config is stored verbatim on the agent."""
    cfg = PpoConfig(obs_dim=17, n_assets=3, hidden_dim=8, total_timesteps=64, n_steps=32)
    agent = PpoAgent(cfg)
    assert agent.config is cfg


@pytest.mark.unit
def test_policy_scores_before_training_raises_validation_error() -> None:
    """``policy_scores`` on an untrained agent raises a clean ``ValidationError``."""
    agent = PpoAgent(PpoConfig(obs_dim=10, n_assets=3))
    with pytest.raises(ValidationError, match="no fitted policy"):
        agent.policy_scores(np.zeros((2, 10)))


@pytest.mark.unit
def test_export_onnx_before_training_raises_validation_error(tmp_path: Path) -> None:
    """``export_onnx`` on an untrained agent raises a clean ``ValidationError``."""
    agent = PpoAgent()
    with pytest.raises(ValidationError, match="no fitted policy"):
        agent.export_onnx(tmp_path / "policy.onnx")


@pytest.mark.unit
def test_export_onnx_rejects_bad_rtol_before_importing_torch(tmp_path: Path) -> None:
    """A non-positive / non-finite ``rtol`` is rejected before any torch import."""
    agent = PpoAgent()
    for bad in (0.0, -1e-4, float("nan"), float("inf")):
        with pytest.raises(ValidationError, match="rtol"):
            agent.export_onnx(tmp_path / "policy.onnx", rtol=bad)


@pytest.mark.unit
def test_train_rejects_negative_seed() -> None:
    """A negative training seed is rejected before any heavy import."""
    agent = PpoAgent()
    with pytest.raises(ValueError, match="seed"):
        agent.train(env=object(), seed=-1)


@pytest.mark.unit
@pytest.mark.skipif(_HAS_TRAIN, reason="[train] extra (torch+sb3) installed: train would proceed")
def test_train_without_train_extra_raises_not_implemented() -> None:
    """Without the ``[train]`` extra, ``train`` raises ``NotImplementedError`` (path unavailable).

    The lean serve image / CI installs no torch / sb3, so the offline training path is
    simply not available; the agent surfaces this as a ``NotImplementedError`` (the
    serve path never invokes training).
    """
    agent = PpoAgent(PpoConfig(obs_dim=10, n_assets=3))
    with pytest.raises(NotImplementedError, match="train"):
        agent.train(env=object(), seed=7)
