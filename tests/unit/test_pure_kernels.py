"""Unit tests for the PURE honesty kernels (DSR/PSR, DM, PBO, seed-lottery, metrics)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rlallocator._exceptions import ValidationError
from rlallocator.evaluation.diebold_mariano import diebold_mariano, dm_favours_model
from rlallocator.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from rlallocator.evaluation.metrics import (
    andrews_lag,
    hac_standard_error,
    max_drawdown,
    net_pnl,
    oos_sharpe,
    turnover,
)
from rlallocator.evaluation.pbo import probability_of_backtest_overfitting
from rlallocator.evaluation.seed_lottery import seed_lottery, variance_of_seed_sharpes


@pytest.mark.unit
def test_psr_in_unit_interval() -> None:
    """The PSR is a probability in [0, 1] and rises with the observed Sharpe."""
    low = probabilistic_sharpe_ratio(0.01, n_obs=500)
    high = probabilistic_sharpe_ratio(0.2, n_obs=500)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


@pytest.mark.unit
def test_psr_rejects_excess_kurtosis_footgun() -> None:
    """Full (non-excess) kurtosis: a Gaussian uses (3-1)/4; a near-zero value is invalid."""
    # kurtosis < 1 with a large Sharpe can drive the bracket variance non-positive.
    with pytest.raises(ValidationError):
        probabilistic_sharpe_ratio(5.0, n_obs=100, kurtosis=0.0)


@pytest.mark.unit
def test_dsr_is_non_increasing_in_n_trials() -> None:
    """The Deflated Sharpe is non-increasing in the multiplicity count n_trials."""
    kwargs = {"n_obs": 500, "variance_of_trial_sharpes": 0.02}
    dsr_few = deflated_sharpe_ratio(0.1, n_trials=1, **kwargs)
    dsr_many = deflated_sharpe_ratio(0.1, n_trials=64, **kwargs)
    assert dsr_few >= dsr_many


@pytest.mark.unit
def test_dsr_is_a_probability() -> None:
    """The DSR lies in [0, 1] (it is a probability, gated at a confidence level)."""
    dsr = deflated_sharpe_ratio(0.15, n_obs=500, n_trials=5, variance_of_trial_sharpes=0.01)
    assert 0.0 <= dsr <= 1.0


@pytest.mark.unit
def test_dm_sign_convention_positive_favours_rl() -> None:
    """A uniformly-better RL series yields a POSITIVE DM statistic."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.01, size=400)
    # A higher mean net return PLUS independent noise (so the differential has dispersion).
    rl = base + 0.002 + rng.normal(0.0, 0.001, size=400)
    stat, pvalue = diebold_mariano(rl, base)
    assert stat > 0.0
    assert 0.0 <= pvalue <= 1.0
    assert dm_favours_model(stat, pvalue, alpha=0.05)


@pytest.mark.unit
def test_dm_identical_series_is_null() -> None:
    """Pointwise-identical series yield the honest null (stat 0, pvalue 1)."""
    x = np.linspace(-0.01, 0.01, 50)
    assert diebold_mariano(x, x) == (0.0, 1.0)


@pytest.mark.unit
def test_pbo_low_when_one_config_dominates() -> None:
    """A single genuinely-best config across all bars yields a low PBO."""
    rng = np.random.default_rng(1)
    n_obs, n_configs = 480, 6
    perf = rng.normal(0.0, 0.01, size=(n_obs, n_configs))
    perf[:, 0] += 0.004  # config 0 is uniformly best in- and out-of-sample.
    result = probability_of_backtest_overfitting(perf, n_splits=8)
    assert 0.0 <= result.pbo <= 1.0
    assert result.pbo < 0.5


@pytest.mark.unit
def test_pbo_high_for_pure_noise() -> None:
    """Pure-noise configs (no real edge) yield a PBO near / above 0.5 (overfit)."""
    rng = np.random.default_rng(2)
    perf = rng.normal(0.0, 0.01, size=(480, 6))
    result = probability_of_backtest_overfitting(perf, n_splits=8)
    assert result.pbo >= 0.3  # noise: the IS-best is not reliably the OOS winner.


@pytest.mark.unit
def test_pbo_rejects_odd_splits() -> None:
    """PBO requires an even number of splits."""
    perf = np.random.default_rng(3).normal(size=(100, 4))
    with pytest.raises(ValidationError):
        probability_of_backtest_overfitting(perf, n_splits=7)


@pytest.mark.unit
def test_seed_lottery_lower_bound_below_median() -> None:
    """The seed-lottery lower bound is below the median for a dispersed set."""
    sharpes = np.array([-0.3, 0.1, 0.4, -0.1, 0.2], dtype="float64")
    result = seed_lottery(sharpes, seed=7)
    assert result.sharpe_lo <= result.median_sharpe <= result.sharpe_hi
    assert result.n_seeds == 5
    assert math.isclose(result.median_sharpe, float(np.median(sharpes)))


@pytest.mark.unit
def test_variance_of_seed_sharpes_matches_numpy() -> None:
    """The cross-seed variance equals numpy's ddof=1 variance."""
    sharpes = np.array([0.1, 0.2, -0.1, 0.05], dtype="float64")
    assert math.isclose(
        variance_of_seed_sharpes(sharpes), float(np.var(sharpes, ddof=1)), rel_tol=1e-12
    )


@pytest.mark.unit
def test_oos_sharpe_flat_series_is_nan() -> None:
    """A flat (zero-dispersion) net-return series has an undefined (NaN) Sharpe."""
    assert math.isnan(oos_sharpe(np.zeros(50)))


@pytest.mark.unit
def test_max_drawdown_non_positive() -> None:
    """Max drawdown is <= 0, and a monotone-up series has zero drawdown."""
    up = np.full(20, 0.01)
    assert max_drawdown(up) == 0.0
    down = np.array([0.1, -0.5, 0.0])
    assert max_drawdown(down) < 0.0


@pytest.mark.unit
def test_turnover_of_weight_path_l1() -> None:
    """Turnover sums per-bar L1 weight changes against an all-cash open."""
    path = np.array([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]], dtype="float64")
    # bar0: |0.5-0| + |0.5-0| = 1.0; bar1: 0; bar2: |1-0.5| + |0-0.5| = 1.0 -> 2.0
    assert math.isclose(turnover(path), 2.0)


@pytest.mark.unit
def test_net_pnl_compounds() -> None:
    """Net PnL is the compounded total return minus one."""
    rets = np.array([0.1, -0.05, 0.02])
    assert math.isclose(net_pnl(rets), float(np.prod(1.0 + rets) - 1.0))


@pytest.mark.unit
def test_andrews_lag_and_hac_se_positive() -> None:
    """The Andrews lag is positive and the HAC SE of a noisy mean is positive."""
    assert andrews_lag(252) > 0
    se = hac_standard_error(np.random.default_rng(4).normal(size=200))
    assert se > 0.0
