"""Parity: the pure kernels (DSR, DM, PBO, baseline weights, metrics) vs hand references."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rlallocator._constants import PERIODS_PER_YEAR
from rlallocator.agents.baselines import markowitz_weights, risk_parity_weights, sample_covariance
from rlallocator.evaluation.diebold_mariano import diebold_mariano
from rlallocator.evaluation.dsr import probabilistic_sharpe_ratio
from rlallocator.evaluation.metrics import max_drawdown, oos_sharpe, turnover


@pytest.mark.parity
def test_psr_against_closed_form_gaussian() -> None:
    """PSR matches Phi(SR*sqrt(n-1)/sqrt(1+(k-1)/4*SR^2)) (skew 0, full kurtosis 3)."""
    sr, n = 0.05, 250
    variance = 1.0 + 0.25 * (3.0 - 1.0) * sr * sr  # 1 - skew*SR + (k-1)/4*SR^2, skew=0
    z = sr * math.sqrt(n - 1) / math.sqrt(variance)
    expected = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    got = probabilistic_sharpe_ratio(sr, n_obs=n)
    assert math.isclose(got, expected, abs_tol=1e-9)


@pytest.mark.parity
def test_oos_sharpe_against_hand_reference() -> None:
    """Annualized Sharpe matches mean/std*sqrt(ppy) on a known series."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=500)
    expected = float(np.mean(rets)) / float(np.std(rets, ddof=1)) * math.sqrt(PERIODS_PER_YEAR)
    assert math.isclose(oos_sharpe(rets), expected, rel_tol=1e-12)


@pytest.mark.parity
def test_dm_statistic_against_hand_reference() -> None:
    """The DM statistic equals d_bar / HAC_SE(d) on a known differential."""
    from rlallocator.evaluation.metrics import hac_standard_error

    rng = np.random.default_rng(1)
    rl = rng.normal(0.0, 0.01, size=300)
    base = rng.normal(0.0, 0.01, size=300)
    diff = rl - base
    expected_stat = float(np.mean(diff)) / hac_standard_error(diff)
    stat, _ = diebold_mariano(rl, base)
    assert math.isclose(stat, expected_stat, rel_tol=1e-9)


@pytest.mark.parity
def test_markowitz_minvar_against_two_asset_closed_form() -> None:
    """Two-asset minimum-variance matches the closed-form long-only solution."""
    # Uncorrelated assets: w_i proportional to 1/variance_i.
    cov = np.diag([0.04, 0.01])
    w = markowitz_weights(cov)
    inv = np.array([1.0 / 0.04, 1.0 / 0.01])
    expected = inv / inv.sum()
    assert np.allclose(w, expected, atol=1e-9)


@pytest.mark.parity
def test_risk_parity_two_asset_inverse_vol() -> None:
    """Two-asset risk-parity (diagonal) equals inverse-volatility."""
    cov = np.diag([0.04, 0.01])
    w = risk_parity_weights(cov)
    inv_vol = np.array([1.0 / 0.2, 1.0 / 0.1])
    expected = inv_vol / inv_vol.sum()
    assert np.allclose(w, expected, atol=1e-6)


@pytest.mark.parity
def test_max_drawdown_hand_reference() -> None:
    """Max drawdown matches a hand-computed peak-to-trough on a known series."""
    rets = np.array([0.5, -0.5, 0.0])  # wealth 1.5, 0.75, 0.75; dd = 0.75/1.5 - 1 = -0.5
    assert math.isclose(max_drawdown(rets), -0.5)


@pytest.mark.parity
def test_turnover_hand_reference() -> None:
    """Turnover matches a hand-computed L1 sum on a known weight path."""
    path = np.array([[1.0, 0.0], [0.0, 1.0]])
    # bar0: |1| + |0| = 1 vs flat; bar1: |0-1| + |1-0| = 2 -> 3.0
    assert math.isclose(turnover(path), 3.0)


@pytest.mark.parity
def test_sample_covariance_against_numpy() -> None:
    """sample_covariance (minus the ridge) matches numpy's ddof=1 covariance."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 4))
    cov = sample_covariance(x)
    expected = np.cov(x, rowvar=False, ddof=1)
    assert np.allclose(cov, expected, atol=1e-6)
