"""Edge-case + serialization tests for the PURE evaluation kernels.

Complements ``test_pure_kernels`` (happy-path numerics) and
``test_verdict_truth_table`` (the verdict gates) with the validation branches,
the ``to_dict`` JSON contracts, the ``strategy_metrics`` bundle, the
Diebold-Mariano scale-aware degeneracy guard, the seed-lottery single-seed
empirical fallback, the PBO shape/short-sample guards, and the verdict
input-validation rejections — so the evaluation group's own tests fully exercise
its own code (no reliance on other groups' coverage-fill tests).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from rlallocator._exceptions import ValidationError
from rlallocator.evaluation.diebold_mariano import diebold_mariano, dm_favours_model
from rlallocator.evaluation.dsr import deflated_sharpe_ratio
from rlallocator.evaluation.metrics import (
    StrategyMetrics,
    andrews_lag,
    hac_standard_error,
    max_drawdown,
    net_pnl,
    oos_sharpe,
    strategy_metrics,
    turnover,
)
from rlallocator.evaluation.pbo import probability_of_backtest_overfitting
from rlallocator.evaluation.seed_lottery import (
    SeedLotteryResult,
    seed_lottery,
    variance_of_seed_sharpes,
)
from rlallocator.evaluation.verdict import Verdict, derive_verdict

# --------------------------------------------------------------------------- #
# metrics: the coercion boundary + strategy_metrics bundle + to_dict           #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_metrics_reject_empty_series() -> None:
    """Every scalar metric rejects an empty per-bar series."""
    empty = np.array([], dtype="float64")
    for fn in (oos_sharpe, max_drawdown, net_pnl):
        with pytest.raises(ValidationError):
            fn(empty)


@pytest.mark.unit
def test_metrics_reject_non_finite_series() -> None:
    """A NaN/inf in the net-return series is rejected, never propagated."""
    for bad in (np.array([0.01, np.nan]), np.array([0.01, np.inf])):
        with pytest.raises(ValidationError):
            oos_sharpe(bad)


@pytest.mark.unit
def test_oos_sharpe_single_observation_is_nan() -> None:
    """A single-observation series has no dispersion estimate -> NaN Sharpe."""
    assert math.isnan(oos_sharpe(np.array([0.01])))


@pytest.mark.unit
def test_oos_sharpe_risk_free_subtracted() -> None:
    """The per-bar risk_free shifts the numerator mean (lowers the Sharpe)."""
    rets = np.array([0.01, 0.02, -0.005, 0.015, 0.0])
    high = oos_sharpe(rets, risk_free=0.0)
    low = oos_sharpe(rets, risk_free=0.005)
    assert high > low


@pytest.mark.unit
def test_turnover_reshapes_1d_path() -> None:
    """A 1-D single-asset weight path is reshaped to a column and L1-summed."""
    # flat opens to 0.5 (|0.5|), then 0.5->0.2 (|0.3|) -> 0.8
    assert math.isclose(turnover(np.array([0.5, 0.2])), 0.8)


@pytest.mark.unit
def test_turnover_with_initial_weights() -> None:
    """A non-flat opening book changes the first-bar L1 turnover."""
    path = np.array([[0.5, 0.5]])
    # opens already AT [0.5, 0.5] -> zero first-bar turnover.
    assert math.isclose(turnover(path, initial_weights=np.array([0.5, 0.5])), 0.0)


@pytest.mark.unit
def test_turnover_rejects_empty_and_misshaped() -> None:
    """Turnover rejects an empty path, a non-finite path, and a misshaped opener."""
    with pytest.raises(ValidationError):
        turnover(np.empty((0, 3)))
    with pytest.raises(ValidationError):
        turnover(np.array([[0.5, np.nan]]))
    with pytest.raises(ValidationError):
        turnover(np.array([[0.5, 0.5]]), initial_weights=np.array([1.0, 0.0, 0.0]))


@pytest.mark.unit
def test_strategy_metrics_bundle_and_to_dict() -> None:
    """strategy_metrics assembles the four scalars; to_dict is JSON-serializable."""
    rets = np.array([0.01, -0.02, 0.015, 0.0, 0.005])
    path = np.array([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 1.0], [0.0, 1.0]], dtype="float64")
    bundle = strategy_metrics(rets, path)
    assert isinstance(bundle, StrategyMetrics)
    assert bundle.n_bars == 5
    assert math.isclose(bundle.oos_sharpe, oos_sharpe(rets), rel_tol=1e-12) or math.isnan(
        bundle.oos_sharpe
    )
    assert math.isclose(bundle.max_drawdown, max_drawdown(rets))
    assert math.isclose(bundle.turnover, turnover(path))
    assert math.isclose(bundle.net_pnl, net_pnl(rets))

    d = bundle.to_dict()
    assert set(d) == {"oos_sharpe", "max_drawdown", "turnover", "net_pnl", "n_bars"}
    # Round-trips through JSON (the API serializes this).
    assert json.loads(json.dumps(d))["n_bars"] == 5


@pytest.mark.unit
def test_strategy_metrics_accepts_1d_single_asset_path() -> None:
    """A 1-D single-asset weight path is reshaped to a column before alignment."""
    rets = np.array([0.01, -0.02, 0.015])
    weights_1d = np.array([0.5, 0.5, 1.0])  # one asset, three bars.
    bundle = strategy_metrics(rets, weights_1d)
    assert bundle.n_bars == 3
    assert math.isclose(bundle.turnover, turnover(weights_1d))


@pytest.mark.unit
def test_strategy_metrics_length_mismatch_rejected() -> None:
    """net_returns rows must equal the weight-path rows."""
    with pytest.raises(ValidationError):
        strategy_metrics(np.array([0.01, 0.02]), np.array([[1.0, 0.0]]))


@pytest.mark.unit
def test_hac_se_rejects_too_few_and_negative_lag() -> None:
    """HAC SE needs >= 2 finite obs and a non-negative lag."""
    with pytest.raises(ValidationError):
        hac_standard_error(np.array([0.01]))
    with pytest.raises(ValidationError):
        hac_standard_error(np.array([0.01, 0.02, 0.03]), lag=-1)


@pytest.mark.unit
def test_hac_se_explicit_zero_lag_is_plain_se() -> None:
    """lag=0 drops all autocovariance terms -> sqrt(gamma0 / T)."""
    x = np.random.default_rng(5).normal(size=120)
    centred = x - x.mean()
    gamma0 = float(centred @ centred / x.size)
    assert math.isclose(hac_standard_error(x, lag=0), math.sqrt(gamma0 / x.size), rel_tol=1e-12)


@pytest.mark.unit
def test_andrews_lag_rejects_non_positive_t() -> None:
    """The Andrews lag rule requires a positive sample size."""
    with pytest.raises(ValidationError):
        andrews_lag(0)


# --------------------------------------------------------------------------- #
# diebold_mariano: pairing guards + the scale-aware degeneracy guard           #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_dm_rejects_length_mismatch_and_non_finite() -> None:
    """The DM pair coercion rejects empty, mismatched, and non-finite entries.

    Exercises every branch of the shared pair-coercion boundary: empty inputs,
    a length mismatch, a non-finite RL series, and a non-finite baseline series.
    """
    with pytest.raises(ValidationError):
        diebold_mariano(np.array([], dtype="float64"), np.array([], dtype="float64"))
    with pytest.raises(ValidationError):
        diebold_mariano(np.array([0.01, 0.02]), np.array([0.01, 0.02, 0.03]))
    with pytest.raises(ValidationError):
        diebold_mariano(np.array([0.01, np.nan]), np.array([0.01, 0.02]))
    with pytest.raises(ValidationError):
        diebold_mariano(np.array([0.01, 0.02]), np.array([0.01, np.inf]))


@pytest.mark.unit
def test_dm_requires_at_least_two_bars() -> None:
    """A single-bar differential cannot support the asymptotic DM statistic."""
    with pytest.raises(ValidationError):
        diebold_mariano(np.array([0.02]), np.array([0.01]))


@pytest.mark.unit
def test_dm_constant_nonzero_differential_is_degenerate() -> None:
    """A uniformly-better but zero-variance differential has an undefined DM stat.

    The scale-aware guard (range vs a magnitude-scaled tolerance) must catch a
    constant non-zero differential rather than dividing by a ~0 HAC SE.
    """
    rl = np.full(40, 0.03)
    base = np.full(40, 0.01)  # diff is a constant +0.02 (no dispersion).
    with pytest.raises(ValidationError):
        diebold_mariano(rl, base)


@pytest.mark.unit
def test_dm_favours_model_requires_significance_and_sign() -> None:
    """dm_favours_model is True only when significant AND positively signed."""
    assert dm_favours_model(3.0, 0.001) is True
    assert dm_favours_model(-3.0, 0.001) is False  # significant but wrong sign
    assert dm_favours_model(3.0, 0.20) is False  # right sign but insignificant


# --------------------------------------------------------------------------- #
# pbo: shape / short-sample guards + to_dict                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_pbo_rejects_non_2d_and_single_config() -> None:
    """PBO requires a 2-D (T, N) matrix with N >= 2 configurations."""
    with pytest.raises(ValidationError):
        probability_of_backtest_overfitting(np.zeros(50), n_splits=4)
    with pytest.raises(ValidationError):
        probability_of_backtest_overfitting(np.zeros((50, 1)), n_splits=4)


@pytest.mark.unit
def test_pbo_rejects_short_sample_and_non_finite() -> None:
    """PBO rejects T < n_splits, too-thin IS/OOS halves, and non-finite perf."""
    with pytest.raises(ValidationError):
        probability_of_backtest_overfitting(np.zeros((4, 3)), n_splits=8)
    # T=3, n_splits=2 => blocks [2, 1]; the smaller half (1 block) carries 1 row
    # (< 2), so the per-block ddof=1 Sharpe ranking is undefined.
    with pytest.raises(ValidationError):
        probability_of_backtest_overfitting(np.zeros((3, 3)), n_splits=2)
    bad = np.random.default_rng(0).normal(size=(64, 3))
    bad[0, 0] = np.nan
    with pytest.raises(ValidationError):
        probability_of_backtest_overfitting(bad, n_splits=4)


@pytest.mark.unit
def test_pbo_to_dict_round_trips() -> None:
    """The PBO result serializes to a JSON-safe dict (logits as a float list)."""
    rng = np.random.default_rng(7)
    perf = rng.normal(0.0, 0.01, size=(240, 5))
    result = probability_of_backtest_overfitting(perf, n_splits=6)
    d = result.to_dict()
    assert isinstance(d["logits"], list)
    assert len(d["logits"]) == result.n_partitions
    parsed = json.loads(json.dumps(d))
    assert 0.0 <= parsed["pbo"] <= 1.0


@pytest.mark.unit
def test_pbo_flat_config_never_is_best() -> None:
    """A flat (zero-dispersion) configuration column is handled (Sharpe -> -inf)."""
    rng = np.random.default_rng(9)
    perf = rng.normal(0.0, 0.01, size=(200, 4))
    perf[:, 3] = 0.0  # a perfectly flat configuration.
    result = probability_of_backtest_overfitting(perf, n_splits=8)
    assert 0.0 <= result.pbo <= 1.0


# --------------------------------------------------------------------------- #
# seed_lottery: validation, single-seed fallback, to_dict                      #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_seed_lottery_rejects_empty_and_bad_alpha() -> None:
    """The lottery rejects an empty / non-finite set and an out-of-range alpha."""
    with pytest.raises(ValidationError):
        seed_lottery(np.array([], dtype="float64"))
    with pytest.raises(ValidationError):
        seed_lottery(np.array([0.1, np.nan]))
    with pytest.raises(ValidationError):
        seed_lottery(np.array([0.1, 0.2]), alpha=1.5)
    with pytest.raises(ValidationError):
        seed_lottery(np.array([0.1, 0.2]), n_bootstrap=0)


@pytest.mark.unit
def test_seed_lottery_single_seed_empirical_fallback() -> None:
    """One seed: both bounds collapse onto that seed's Sharpe (no bootstrap)."""
    result = seed_lottery(np.array([0.42]))
    assert result.n_seeds == 1
    assert math.isclose(result.median_sharpe, 0.42)
    assert math.isclose(result.sharpe_lo, 0.42)
    assert math.isclose(result.sharpe_hi, 0.42)
    assert result.sharpe_std == 0.0


@pytest.mark.unit
def test_seed_lottery_to_dict_round_trips() -> None:
    """The lottery result serializes (seed_sharpes as a plain float list)."""
    result = seed_lottery(np.array([-0.1, 0.0, 0.2, 0.3]), seed=7)
    assert isinstance(result, SeedLotteryResult)
    d = result.to_dict()
    assert isinstance(d["seed_sharpes"], list)
    parsed = json.loads(json.dumps(d))
    assert parsed["n_seeds"] == 4


@pytest.mark.unit
def test_seed_lottery_bootstrap_is_deterministic() -> None:
    """The seeded bootstrap is byte-reproducible across calls."""
    sharpes = np.array([-0.2, 0.05, 0.3, 0.1, -0.05])
    a = seed_lottery(sharpes, seed=7)
    b = seed_lottery(sharpes, seed=7)
    assert a.sharpe_lo == b.sharpe_lo
    assert a.sharpe_hi == b.sharpe_hi


@pytest.mark.unit
def test_variance_of_seed_sharpes_requires_two() -> None:
    """A single seed has no dispersion estimate for the DSR variance term."""
    with pytest.raises(ValidationError):
        variance_of_seed_sharpes(np.array([0.1]))


# --------------------------------------------------------------------------- #
# verdict: input validation + to_dict + best-baseline DM wiring                #
# --------------------------------------------------------------------------- #

_PASS = {
    "dm_statistic": 3.0,
    "dm_pvalue": 0.001,
    "deflated_sharpe": 0.99,
    "seed_sharpe_lo": 0.2,
    "pbo": 0.1,
    "n_effective_trials": 5,
}


@pytest.mark.unit
def test_verdict_rejects_non_finite_and_bad_trials() -> None:
    """The verdict rejects non-finite evidence and a non-positive trial count."""
    with pytest.raises(ValidationError):
        derive_verdict(**{**_PASS, "dm_statistic": math.inf})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        derive_verdict(**{**_PASS, "deflated_sharpe": math.nan})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        derive_verdict(**{**_PASS, "seed_sharpe_lo": math.inf})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        derive_verdict(**{**_PASS, "n_effective_trials": 0})  # type: ignore[arg-type]


@pytest.mark.unit
def test_verdict_to_dict_serializes_enum() -> None:
    """The verdict result serializes the enum to its stable string value."""
    d = derive_verdict(**_PASS).to_dict()  # type: ignore[arg-type]
    assert d["verdict"] == Verdict.RL_BEATS_BASELINES.value
    parsed = json.loads(json.dumps(d))
    assert parsed["rl_beats_baselines"] is True
    assert parsed["n_effective_trials"] == 5


@pytest.mark.unit
def test_verdict_alpha_widens_dsr_gate() -> None:
    """A looser alpha lowers the DSR confidence threshold (1 - alpha)."""
    cfg = {**_PASS, "deflated_sharpe": 0.91}
    # alpha=0.05 => gate 0.95 => fails; alpha=0.10 => gate 0.90 => passes.
    assert derive_verdict(**cfg, alpha=0.05).rl_beats_baselines is False  # type: ignore[arg-type]
    assert derive_verdict(**cfg, alpha=0.10).rl_beats_baselines is True  # type: ignore[arg-type]


@pytest.mark.unit
def test_verdict_carries_through_best_baseline_dm_evidence() -> None:
    """The verdict echoes the DM-vs-best-baseline evidence it was given.

    Best-baseline SELECTION happens upstream (serve picks the highest-Sharpe of
    1/N / Markowitz / risk-parity and runs DM against it); the verdict consumes
    that DM p-value verbatim and must surface it in its evidence bundle.
    """
    # A near-miss DM p-value (just at the boundary) drives the NULL and is echoed.
    cfg = {**_PASS, "dm_pvalue": 0.05}  # NOT < alpha => fails the DM gate.
    result = derive_verdict(**cfg)  # type: ignore[arg-type]
    assert result.rl_beats_baselines is False
    assert result.dm_pvalue == 0.05
    assert result.verdict is Verdict.NO_SIGNIFICANT_DIFFERENCE


@pytest.mark.unit
def test_dsr_n_trials_honesty_drives_verdict() -> None:
    """A real edge under N=1 can be deflated below the confidence gate under N=many.

    Wires the DSR n_trials honesty (#seeds x #HP) straight into the verdict: the
    SAME observed Sharpe that clears 0.95 with one trial is deflated under a large
    honest multiplicity, flipping the verdict to the NULL.
    """
    dsr_one = deflated_sharpe_ratio(
        0.18, n_obs=504, n_trials=1, variance_of_trial_sharpes=0.04, skew=0.0, kurtosis=3.0
    )
    dsr_many = deflated_sharpe_ratio(
        0.18, n_obs=504, n_trials=128, variance_of_trial_sharpes=0.04, skew=0.0, kurtosis=3.0
    )
    assert dsr_one > dsr_many  # honest multiplicity deflates the same Sharpe.

    # The single-trial DSR clears 0.95; the deflated many-trial DSR does not, so
    # the SAME observed Sharpe flips the verdict once the honest n_trials is wired in.
    one = derive_verdict(
        dm_statistic=3.0,
        dm_pvalue=0.001,
        deflated_sharpe=dsr_one,
        seed_sharpe_lo=0.2,
        pbo=0.1,
        n_effective_trials=1,
    )
    many = derive_verdict(
        dm_statistic=3.0,
        dm_pvalue=0.001,
        deflated_sharpe=dsr_many,
        seed_sharpe_lo=0.2,
        pbo=0.1,
        n_effective_trials=128,
    )
    assert one.rl_beats_baselines is True
    assert many.rl_beats_baselines is False
