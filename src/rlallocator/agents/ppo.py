"""SB3 PPO allocator wrapper (torch + stable-baselines3, LAZY, the ``[train]`` extra).

TYPED STUB (SCAFFOLD). The OFFLINE training agent: wraps a Stable-Baselines3 PPO
policy trained in the gymnasium-wrapped
:class:`rlallocator.env.portfolio_env.PortfolioEnv`. torch, stable-baselines3, and
gymnasium are imported LAZILY inside the methods (the ``[train]`` extra), so
importing this module pulls in NO torch / sb3 / gymnasium and has no side effects.
This agent is NEVER invoked on the request path — the served policy is the exported
ONNX MLP (:mod:`rlallocator.agents.onnx_policy`).

FALLBACK plan (documented for the implementer): the SB3 ``MlpPolicy`` mixes a value
head + an action-distribution layer into its forward pass, none of which the serve
path needs. Rather than wrestle the full SB3 ``ActorCriticPolicy`` graph through
``torch.onnx.export`` (it is not ONNX-clean), :meth:`export_onnx` will EXTRACT only
the small policy MLP — the shared/pi feature extractor plus the action head (the
per-asset score head whose softmax/projection is the simplex) — into a standalone
plain-torch :class:`~torch.nn.Module` and export THAT, validated 1e-4 against torch.
The honest-null deliverable holds either way; the shipped path is documented in the
README.

Importing this module has no side effects. The method bodies raise
``NotImplementedError`` until the offline training is wired up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlallocator._typing import FloatArray, ObservationMatrix

if TYPE_CHECKING:  # pragma: no cover - typing-only import (torch is the [train] extra)
    import torch

#: The ONNX input tensor name (the per-bar observation: look-back window + weights).
OBSERVATION_INPUT: str = "observation"

#: The ONNX output tensor name (the per-asset score head; projected to the simplex).
ACTION_OUTPUT: str = "asset_scores"

#: The ONNX opset for the LEGACY exporter (``dynamo=False``; the dynamo path needs
#: onnxscript, which is NOT a dependency of this project).
ONNX_OPSET: int = 17


@dataclass(frozen=True, slots=True)
class PpoConfig:
    """Immutable PPO hyper-parameter configuration (defines the ONNX export signature).

    Attributes
    ----------
    obs_dim:
        The observation dimension (flattened look-back window + current weights:
        ``lookback * n_assets + n_assets``).
    n_assets:
        The number of assets (the per-asset score head width; softmax/projected to
        the long-only weight simplex).
    hidden_dim:
        The MLP hidden width (small — the policy is a tiny MLP for clean ONNX).
    n_steps:
        PPO rollout length per update.
    total_timesteps:
        Total training timesteps per seed.
    learning_rate:
        The Adam learning rate.
    gamma:
        The reward discount factor.
    """

    obs_dim: int = 390
    n_assets: int = 6
    hidden_dim: int = 64
    n_steps: int = 2048
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    gamma: float = 0.99

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


class PpoAgent:
    """A thin SB3 PPO wrapper that trains in the gym-wrapped portfolio env (LAZY torch/sb3).

    TYPED STUB (SCAFFOLD). Construction is cheap and import-pure: torch /
    stable-baselines3 / gymnasium are imported only inside :meth:`train` /
    :meth:`export_onnx` / :meth:`policy_scores` (the ``[train]`` extra). The trained
    policy is exported to a small ONNX MLP and served through onnxruntime — torch is
    NEVER imported on the request path. The method bodies raise
    ``NotImplementedError`` until the offline training is implemented.
    """

    def __init__(self, config: PpoConfig | None = None) -> None:
        """Record the config; defer the SB3 model build to :meth:`train`.

        Parameters
        ----------
        config:
            The PPO hyper-parameter config; ``None`` => :class:`PpoConfig` defaults.
        """
        self._config: PpoConfig = config if config is not None else PpoConfig()
        #: The fitted SB3 ``PPO`` model (set by :meth:`train`).
        self._model: Any | None = None
        #: The extracted standalone policy MLP (lazily built from the SB3 policy).
        self._policy_net: torch.nn.Module | None = None

    @property
    def config(self) -> PpoConfig:
        """Return the immutable PPO configuration this agent was built with."""
        return self._config

    @property
    def is_trained(self) -> bool:
        """Return ``True`` once :meth:`train` has produced a fitted policy."""
        return self._policy_net is not None

    def train(self, env: Any, *, seed: int = 7) -> PpoAgent:
        """Train the PPO policy in the gym-wrapped portfolio env for one seed (LAZY torch/sb3).

        LAZY IMPORT: ``stable_baselines3`` / ``torch`` (the ``[train]`` extra) load
        inside this method. Trains a PPO policy on the gymnasium-wrapped
        :class:`PortfolioEnv` for ``config.total_timesteps`` with the given ``seed``,
        then extracts the policy MLP (pi feature extractor + per-asset score head)
        into a standalone plain-torch module ready for ONNX export. NEVER invoked on
        the request path.

        Parameters
        ----------
        env:
            A gymnasium-API env (e.g. ``PortfolioEnv.as_gym_env()``).
        seed:
            The torch / SB3 RNG seed for this training run (one per seed-lottery draw).

        Returns
        -------
        PpoAgent
            ``self``, with a fitted policy.

        Raises
        ------
        NotImplementedError
            Until the offline training is implemented.
        """
        raise NotImplementedError(
            "PpoAgent.train is a scaffold stub; the offline SB3 PPO training "
            "(the [train] extra) is implemented by the train.py author."
        )

    def policy_scores(self, observations: ObservationMatrix) -> FloatArray:
        """Return the fitted policy's per-asset score logits for a batch (LAZY torch).

        The reference output the exported ONNX graph is validated against to 1e-4;
        the scores are projected onto the weight simplex by the env / serve path.

        Parameters
        ----------
        observations:
            A ``(batch, obs_dim)`` observation matrix.

        Returns
        -------
        FloatArray
            The ``(batch, n_assets)`` per-asset score logits.

        Raises
        ------
        NotImplementedError
            Until the offline training is implemented.
        """
        raise NotImplementedError(
            "PpoAgent.policy_scores is a scaffold stub; implemented alongside train()."
        )

    def export_onnx(self, out_path: str | Path, *, rtol: float = 1e-4) -> Path:
        """Export ONLY the policy MLP to ONNX with a 1e-4 torch-vs-ONNX parity gate.

        LAZY IMPORT: ``torch`` (the ``[train]`` extra) and ``onnxruntime`` load inside
        this method. Exports the extracted policy network (obs -> per-asset scores)
        with ``torch.onnx.export(..., dynamo=False)`` and a DYNAMIC batch axis, then
        asserts the torch and ONNX forward passes agree to ``rtol`` (the parity gate).
        The serve path then uses ONLY the ONNX graph (no torch).

        Parameters
        ----------
        out_path:
            Destination ``.onnx`` path (the committed ``artifacts/policy.onnx``).
        rtol:
            Relative tolerance for the torch-vs-ONNX parity assertion (default 1e-4).

        Returns
        -------
        pathlib.Path
            The written ``.onnx`` path.

        Raises
        ------
        NotImplementedError
            Until the offline training / export is implemented.
        """
        raise NotImplementedError(
            "PpoAgent.export_onnx is a scaffold stub; implemented alongside train()."
        )
