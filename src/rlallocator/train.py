"""Offline training pipeline: synthetic -> purged walk-forward -> PPO across seeds -> ONNX + metrics.

TYPED STUB (SCAFFOLD). This is the OFFLINE path (the ``[train]`` extra, torch +
stable-baselines3 + gymnasium). When implemented it will: build the synthetic
multi-asset panel, split it into purged/embargoed walk-forward folds, train the SB3
PPO allocator across N INDEPENDENT seeds, export the median-seed policy network to
ONNX (parity-checked to torch to 1e-4), precompute the per-seed OOS equity curves +
net Sharpe values, summarize the seed lottery + the Diebold-Mariano-vs-best-baseline +
the Deflated Sharpe + the PBO (with the honest ``n_trials = #seeds x #HP configs``),
derive the PURE ``rl_beats_baselines`` verdict, and write a committed
``artifacts/metrics.json`` + ``artifacts/policy.onnx`` (<10MB) + a :class:`RunManifest`.

torch / stable-baselines3 / gymnasium are imported LAZILY inside the per-seed trainer
seam, so importing this module pulls in NO torch / sb3 / gymnasium and has no side
effects. NEVER invoked on the request path. The pure multiplicity helper
:func:`n_effective_trials` IS implemented (it is needed by the serve / verdict path);
:func:`train_pipeline` raises ``NotImplementedError`` until the offline path is wired.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlallocator._exceptions import ValidationError

#: The package's committed-artifacts directory (policy.onnx + metrics.json).
_DEFAULT_ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
#: The hyper-parameter configurations swept per seed. The honest multiplicity
#: ``n_effective_trials`` is ``#seeds x #HP configs`` — the seed lottery AND the HP
#: grid both count as selection trials (never undercount).
_N_HP_CONFIGS: int = 1


def n_effective_trials(n_seeds: int) -> int:
    """Return the honest DSR multiplicity ``n_seeds x #HP configs``.

    The Deflated-Sharpe benchmark must count the FULL explored configuration grid so
    the selection bias of trying many seeds x hyper-parameters is paid for. This is
    the single source of truth committed to ``metrics.json`` and mirrored by the serve
    fallback. The seed lottery is itself a selection-trial set — undercounting it
    (e.g. counting only HP configs) would inflate the DSR dishonestly.

    Parameters
    ----------
    n_seeds:
        The number of independent training seeds.

    Returns
    -------
    int
        ``n_seeds * _N_HP_CONFIGS``.

    Raises
    ------
    ValidationError
        If ``n_seeds < 1``.
    """
    if n_seeds < 1:
        raise ValidationError(f"n_effective_trials: n_seeds must be >= 1, got {n_seeds}.")
    return n_seeds * _N_HP_CONFIGS


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable summary of an offline training run.

    Attributes
    ----------
    policy_path:
        Path to the exported committed ``policy.onnx``.
    metrics_path:
        Path to the committed ``metrics.json``.
    n_effective_trials:
        The FULL multiplicity count (#seeds x #HP configs) for the DSR.
    rl_beats_baselines:
        The PURE verdict at train time (expected ``False`` on the factor-regime null).
    manifest:
        The reproducibility manifest dict (git SHA, dirty flag, config hash, seed).
    """

    policy_path: str
    metrics_path: str
    n_effective_trials: int
    rl_beats_baselines: bool
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return asdict(self)


def train_pipeline(
    *,
    n_assets: int = 6,
    n_seeds: int = 5,
    lookback: int = 64,
    cost_bps: float = 10.0,
    episode_len: int = 252,
    n_obs: int = 2000,
    n_folds: int = 4,
    kind: str = "factor_regime",
    seed: int = 7,
    artifacts_dir: str | Path | None = None,
) -> TrainResult:
    """Run the offline train -> ONNX-export -> seed-lottery -> metrics pipeline end-to-end.

    TYPED STUB (SCAFFOLD). When implemented it lazily imports torch / sb3 / gymnasium /
    onnx (the ``[train]`` extra), builds the synthetic panel, splits it into
    purged/embargoed walk-forward folds (:func:`rlallocator.walk_forward.make_folds`),
    trains the PPO allocator across ``n_seeds`` seeds with each FROZEN policy scored on
    the OOS folds through the pure-numpy vectorized backtester (NO test-time learning),
    exports the median-seed policy to ONNX (1e-4 parity gate), summarizes the seed
    lottery + DM vs. the best baseline + DSR + PBO (``n_trials = n_seeds x #HP``),
    derives the pure verdict, and writes the committed artifacts + ``metrics.json`` +
    manifest.

    Parameters
    ----------
    n_assets:
        Number of assets in the basket.
    n_seeds:
        Number of independent training seeds (the seed lottery).
    lookback:
        Observation look-back window length.
    cost_bps:
        Per-side turnover cost in basis points (IDENTICAL in train and eval).
    episode_len:
        Max bars per training episode.
    n_obs:
        Number of synthetic bars to generate.
    n_folds:
        Number of purged walk-forward folds.
    kind:
        Synthetic DGP: ``"factor_regime"`` (honest-null default), ``"learnable_edge"``
        (the sanity fixture), or ``"pure_noise"`` (the strict null).
    seed:
        Master RNG seed for the synthetic panel + the seed-substream spawning.
    artifacts_dir:
        Output directory for ``policy.onnx`` + ``metrics.json``; ``None`` => the
        package's ``artifacts/`` directory.

    Returns
    -------
    TrainResult
        Paths, multiplicity count, and the (expected ``False``) honest verdict.

    Raises
    ------
    NotImplementedError
        Until the offline training path is implemented.
    ValidationError
        If the request is invalid (bad lookback / n_seeds / kind).
    """
    if n_seeds < 1:
        raise ValidationError(f"train_pipeline: n_seeds must be >= 1, got {n_seeds}.")
    if lookback < 1:
        raise ValidationError(f"train_pipeline: lookback must be >= 1, got {lookback}.")
    raise NotImplementedError(
        "train_pipeline is a scaffold stub; the offline SB3 PPO training + ONNX export "
        "(the [train] extra) is implemented by the train.py author. The pure "
        "orchestration seam (walk-forward folds, seed lottery, DSR/PBO, verdict) is "
        "already wired into the evaluation kernels and serve.run_allocation."
    )
