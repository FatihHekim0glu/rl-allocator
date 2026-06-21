"""Unit tests for the pure-numpy allocation baselines (1/N, Markowitz, risk-parity)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rlallocator._exceptions import ValidationError
from rlallocator._validation import is_simplex
from rlallocator.agents.baselines import (
    BASELINE_NAMES,
    baseline_weights,
    equal_weight,
    markowitz_weights,
    risk_parity_weights,
    run_baseline,
    sample_covariance,
)


@pytest.mark.unit
def test_equal_weight_is_uniform_simplex() -> None:
    """1/N weights are uniform and a valid simplex."""
    w = equal_weight(5)
    assert np.allclose(w, 0.2)
    assert is_simplex(w)


@pytest.mark.unit
def test_all_baselines_are_valid_simplices(synthetic_panel: pd.DataFrame) -> None:
    """Every baseline weight vector is a valid long-only simplex on the panel."""
    train = synthetic_panel.to_numpy()
    for name in BASELINE_NAMES:
        w = baseline_weights(name, train)
        assert is_simplex(w), name


@pytest.mark.unit
def test_markowitz_minvar_beats_equal_weight_variance() -> None:
    """The minimum-variance portfolio has variance <= equal-weight (its objective)."""
    rng = np.random.default_rng(0)
    # A panel with heterogeneous asset variances.
    cov_true = np.diag([0.04, 0.01, 0.0025, 0.09])
    returns = rng.multivariate_normal(np.zeros(4), cov_true, size=600)
    cov = sample_covariance(returns)
    w_mv = markowitz_weights(cov)
    w_1n = equal_weight(4)
    var_mv = float(w_mv @ cov @ w_mv)
    var_1n = float(w_1n @ cov @ w_1n)
    assert var_mv <= var_1n + 1e-12
    assert is_simplex(w_mv)


@pytest.mark.unit
def test_risk_parity_diagonal_is_inverse_vol() -> None:
    """With a diagonal covariance, risk-parity reduces to inverse-volatility weights."""
    cov = np.diag([0.04, 0.01, 0.0025])  # vols 0.2, 0.1, 0.05
    w = risk_parity_weights(cov)
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    expected = inv_vol / inv_vol.sum()
    assert np.allclose(w, expected, atol=1e-6)
    assert is_simplex(w)


@pytest.mark.unit
def test_risk_parity_equalizes_risk_contributions() -> None:
    """Risk-parity equalizes per-asset risk contributions w_i (Sigma w)_i."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(500, 4))
    cov = sample_covariance(a + 0.1 * a[:, [0]])  # induce correlation
    w = risk_parity_weights(cov)
    rc = w * (cov @ w)
    assert np.allclose(rc, rc.mean(), rtol=1e-3, atol=1e-6)


@pytest.mark.unit
def test_run_baseline_scores_through_backtester(synthetic_panel: pd.DataFrame) -> None:
    """run_baseline scores a baseline on OOS through the shared backtester, train-only cov."""
    arr = synthetic_panel.to_numpy()
    train, oos = arr[:400], arr[400:]
    result = run_baseline("markowitz", train, oos, cost_bps=10.0)
    assert result.n_bars == oos.shape[0] - 1
    assert result.net_returns.size == result.n_bars
    assert result.turnover >= 0.0
    assert math.isfinite(result.net_pnl)


@pytest.mark.unit
def test_baseline_weights_rejects_unknown_name(synthetic_panel: pd.DataFrame) -> None:
    """An unknown baseline name is rejected."""
    with pytest.raises(ValidationError):
        baseline_weights("momentum", synthetic_panel.to_numpy())
