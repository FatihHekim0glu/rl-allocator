"""Regression: the learnable_edge SANITY fixture + the honest-null guard.

- SANITY (anti-vacuous-null): on ``learnable_edge`` an oracle allocator that tilts
  toward the premium asset beats 1/N net of costs — the env + accounting machinery
  actually capture a real edge, so the null is honest, NOT vacuous.
- HONEST NULL: on the factor-regime panel and on pure_noise, NO static re-weighting of
  the baselines reliably beats 1/N net of costs after the purged walk-forward — the
  documented honest-NULL outcome.

These pin the deliverable: the machinery works (sanity) AND the agent does not get a
free edge on the null (honesty).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlallocator.agents.baselines import run_baseline
from rlallocator.env.backtester import vectorized_backtest
from rlallocator.evaluation.metrics import oos_sharpe
from rlallocator.walk_forward import make_folds
from tests.conftest import LEARNABLE_EDGE_PREMIUM_ASSET


def _oracle_tilt_path(n_oos: int, n_assets: int, premium_asset: int) -> np.ndarray:
    """A weight path fully tilted toward the premium asset (the oracle allocation)."""
    w = np.zeros(n_assets, dtype="float64")
    w[premium_asset] = 1.0
    return np.tile(w, (n_oos, 1))


@pytest.mark.regression
def test_learnable_edge_oracle_tilt_beats_equal_weight(learnable_edge: pd.DataFrame) -> None:
    """On learnable_edge, tilting toward the premium asset beats 1/N net of costs (sanity)."""
    arr = learnable_edge.to_numpy()
    train, oos = arr[:400], arr[400:]
    n_assets = arr.shape[1]

    tilt_path = _oracle_tilt_path(oos.shape[0], n_assets, LEARNABLE_EDGE_PREMIUM_ASSET)
    tilt_net = vectorized_backtest(oos, tilt_path, cost_bps=10.0).net_returns
    one_n_net = run_baseline("equal_weight", train, oos, cost_bps=10.0).net_returns

    tilt_sharpe = oos_sharpe(tilt_net)
    one_n_sharpe = oos_sharpe(one_n_net)
    assert tilt_sharpe > one_n_sharpe, (
        "the machinery must capture a REAL premium — the null is honest, not vacuous."
    )


@pytest.mark.regression
def test_honest_null_no_baseline_dominates_1n_walk_forward(synthetic_panel: pd.DataFrame) -> None:
    """On the factor-regime null, no baseline crushes 1/N across the purged folds."""
    arr = synthetic_panel.to_numpy()
    folds = make_folds(arr.shape[0], lookback=16, n_folds=4)
    sharpes: dict[str, list[float]] = {"equal_weight": [], "markowitz": [], "risk_parity": []}
    for fold in folds:
        train = arr[fold.train_start : fold.train_end]
        oos = arr[fold.test_start : fold.test_end]
        for name in sharpes:
            net = run_baseline(name, train, oos, cost_bps=10.0).net_returns
            s = oos_sharpe(net)
            sharpes[name].append(s if np.isfinite(s) else 0.0)

    one_n = float(np.mean(sharpes["equal_weight"]))
    best_estimated = max(
        float(np.mean(sharpes["markowitz"])), float(np.mean(sharpes["risk_parity"]))
    )
    # The estimated baselines do not get a large free edge over 1/N on the null: the
    # gap is small (the honest-NULL shape — no allocation reliably dominates).
    assert best_estimated - one_n < 1.0


@pytest.mark.regression
def test_pure_noise_oos_sharpes_straddle_zero(pure_noise: pd.DataFrame) -> None:
    """On pure noise the baseline OOS Sharpes are small and not reliably positive."""
    arr = pure_noise.to_numpy()
    train, oos = arr[:400], arr[400:]
    sharpe = oos_sharpe(run_baseline("equal_weight", train, oos, cost_bps=10.0).net_returns)
    assert abs(sharpe) < 2.0  # no large, reliable edge on white noise.
