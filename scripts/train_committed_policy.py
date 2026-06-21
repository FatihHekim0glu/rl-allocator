"""One-off offline trainer: produce the committed ONNX policy + metrics.json.

Runs the offline ``train_pipeline`` with a REAL SB3 PPO trainer (the ``[train]`` extra:
torch + stable-baselines3 + gymnasium) across N seeds on the synthetic factor-regime
panel, exporting the median-seed policy to ONNX (<10MB, 1e-4 torch-vs-ONNX parity) and
writing ``src/rlallocator/artifacts/{policy.onnx,metrics.json}``. The honest-NULL verdict
(``rl_beats_baselines=False``) is the expected outcome. NOT part of the request path —
the served container loads the committed ONNX via onnxruntime only.

Usage (offline): ``uv run python scripts/train_committed_policy.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rlallocator.agents.ppo import PpoAgent, PpoConfig
from rlallocator.env.portfolio_env import PortfolioEnv, PortfolioEnvConfig
from rlallocator.train import TrainedPolicy, train_pipeline

#: A tractable-but-genuine PPO budget for the committed artifact (real training, not a
#: toy): a small policy MLP trained for a modest number of timesteps per seed x fold.
_TOTAL_TIMESTEPS: int = 20_000
_HIDDEN_DIM: int = 64
_N_STEPS: int = 512


def _committed_trainer(
    train_returns: np.ndarray,
    *,
    obs_dim: int,
    n_assets: int,
    lookback: int,
    cost_bps: float,
    episode_len: int,
    seed: int,
) -> TrainedPolicy:
    """The real SB3 PPO trainer with the committed-artifact budget (torch + sb3 + gymnasium)."""
    config = PpoConfig(
        obs_dim=obs_dim,
        n_assets=n_assets,
        hidden_dim=_HIDDEN_DIM,
        n_steps=_N_STEPS,
        total_timesteps=_TOTAL_TIMESTEPS,
    )
    env = PortfolioEnv(
        train_returns,
        PortfolioEnvConfig(lookback=lookback, cost_bps=cost_bps),
        episode_len=episode_len,
    )
    agent = PpoAgent(config)
    agent.train(env.as_gym_env(), seed=seed)
    return agent


def main() -> None:
    """Train the committed policy + metrics into the package artifacts directory."""
    result = train_pipeline(
        n_assets=6,
        n_seeds=5,
        lookback=64,
        cost_bps=10.0,
        episode_len=252,
        n_obs=2000,
        n_folds=4,
        kind="factor_regime",
        seed=7,
        trainer=_committed_trainer,
    )
    policy = Path(result.policy_path)
    size_mb = policy.stat().st_size / 1e6 if policy.is_file() else 0.0
    print("offline training complete")
    print(f"  policy:  {result.policy_path} ({size_mb:.3f} MB)")
    print(f"  metrics: {result.metrics_path}")
    print(f"  n_effective_trials: {result.n_effective_trials}")
    print(f"  seed_sharpes: {[round(s, 4) for s in result.seed_sharpes]}")
    print(f"  rl_beats_baselines: {'YES' if result.rl_beats_baselines else 'NO'}")


if __name__ == "__main__":
    main()
