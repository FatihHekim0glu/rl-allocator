"""Unit tests for the pure-numpy allocation baselines (1/N, Markowitz, risk-parity)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rlallocator._exceptions import SingularCovarianceError, ValidationError
from rlallocator._validation import is_simplex
from rlallocator.agents.baselines import (
    BASELINE_NAMES,
    BaselineResult,
    baseline_weight_path,
    baseline_weights,
    equal_weight,
    is_valid_baseline_weights,
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


@pytest.mark.unit
def test_markowitz_diagonal_matches_inverse_variance_hand_reference() -> None:
    """Diagonal-covariance min-variance weights equal the hand-derived w_i proportional to 1/sigma_i^2.

    For a DIAGONAL covariance the unconstrained global minimum-variance solution
    ``w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)`` is interior and reduces to inverse
    variance, ``w_i = (1/sigma_i^2) / sum_j (1/sigma_j^2)``.
    """
    cov = np.diag([0.04, 0.01, 0.0025, 0.16])
    inv_var = 1.0 / np.diag(cov)
    hand = inv_var / inv_var.sum()
    w = markowitz_weights(cov)
    assert np.allclose(w, hand, atol=1e-10)
    assert is_simplex(w)


@pytest.mark.unit
def test_markowitz_two_asset_matches_closed_form_hand_reference() -> None:
    """A 2x2 interior min-variance solution matches the analytic closed form.

    For ``Sigma = [[a, c], [c, b]]`` the unconstrained min-variance weights are
    ``w proportional to (b - c, a - c)`` (since ``Sigma^{-1} 1 proportional to
    (b - c, a - c)``), normalized to sum to one.
    """
    a, b, c = 0.04, 0.09, 0.012
    cov = np.array([[a, c], [c, b]], dtype="float64")
    raw = np.array([b - c, a - c], dtype="float64")
    hand = raw / raw.sum()
    w = markowitz_weights(cov)
    assert np.allclose(w, hand, atol=1e-10)
    assert is_simplex(w)


@pytest.mark.unit
def test_markowitz_active_set_clips_negative_to_long_only_simplex() -> None:
    """When the interior solution goes negative, the active set returns a long-only simplex."""
    # Strong positive correlation between assets 0 and 1 pushes an interior weight
    # negative; the active-set loop must drop it and return non-negative weights.
    cov = np.array(
        [[0.04, 0.039, 0.0], [0.039, 0.04, 0.0], [0.0, 0.0, 0.25]],
        dtype="float64",
    )
    w = markowitz_weights(cov)
    assert is_simplex(w)
    assert bool((w >= 0.0).all())


@pytest.mark.unit
def test_baselines_use_train_window_only_no_lookahead() -> None:
    """Baseline weights depend ONLY on the train window — OOS data never leaks in.

    Running each baseline against two wildly different OOS panels must yield the
    identical applied weight vector (the train-only covariance discipline), proving
    there is no look-ahead from the OOS window into the estimated allocation.
    """
    rng = np.random.default_rng(7)
    train = rng.normal(0.0, 0.01, size=(400, 6))
    oos_calm = rng.normal(0.0, 0.01, size=(120, 6))
    oos_wild = rng.normal(5.0, 2.0, size=(120, 6))  # nothing like the train window
    for name in BASELINE_NAMES:
        expected = baseline_weights(name, train)
        calm = run_baseline(name, train, oos_calm, cost_bps=10.0)
        wild = run_baseline(name, train, oos_wild, cost_bps=10.0)
        # The applied per-bar weight is the fixed train-window vector, OOS-invariant.
        assert np.allclose(calm.weights[0], expected, atol=1e-12), name
        assert np.array_equal(calm.weights[0], wild.weights[0]), name


@pytest.mark.unit
def test_baselines_are_deterministic() -> None:
    """The baselines are pure deterministic maps: identical inputs give bit-identical weights."""
    rng = np.random.default_rng(3)
    train = rng.normal(0.0, 0.01, size=(300, 5))
    for name in BASELINE_NAMES:
        first = baseline_weights(name, train)
        second = baseline_weights(name, train)
        assert np.array_equal(first, second), name
    cov = sample_covariance(train)
    assert np.array_equal(markowitz_weights(cov), markowitz_weights(cov))
    assert np.array_equal(risk_parity_weights(cov), risk_parity_weights(cov))


@pytest.mark.unit
def test_baseline_weight_path_tiles_fixed_train_vector() -> None:
    """The OOS weight path tiles the single fixed train-window weight vector each bar."""
    rng = np.random.default_rng(11)
    train = rng.normal(0.0, 0.01, size=(250, 4))
    w = baseline_weights("risk_parity", train)
    path = baseline_weight_path("risk_parity", train, n_oos_bars=30)
    assert path.shape == (30, 4)
    assert np.allclose(path, w[None, :], atol=1e-12)
    for row in path:
        assert is_simplex(row)


@pytest.mark.unit
def test_baseline_weight_path_rejects_non_positive_bars(synthetic_panel: pd.DataFrame) -> None:
    """A non-positive OOS bar count is rejected."""
    with pytest.raises(ValidationError):
        baseline_weight_path("equal_weight", synthetic_panel.to_numpy(), n_oos_bars=0)


@pytest.mark.unit
def test_run_baseline_rejects_asset_count_mismatch() -> None:
    """A train/OOS asset-count mismatch is rejected (no silent broadcast)."""
    train = np.zeros((50, 4), dtype="float64")
    oos = np.zeros((20, 3), dtype="float64")
    with pytest.raises(ValidationError):
        run_baseline("equal_weight", train, oos, cost_bps=10.0)


@pytest.mark.unit
def test_equal_weight_rejects_non_positive_n() -> None:
    """equal_weight requires at least one asset."""
    with pytest.raises(ValidationError):
        equal_weight(0)


@pytest.mark.unit
def test_sample_covariance_rejects_malformed_and_short_panels() -> None:
    """sample_covariance rejects non-finite, empty, 3-D, and single-observation panels."""
    with pytest.raises(ValidationError):
        sample_covariance(np.array([[np.nan, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValidationError):
        sample_covariance(np.zeros((0, 3)))
    with pytest.raises(ValidationError):
        sample_covariance(np.zeros((2, 2, 2)))
    with pytest.raises(ValidationError):
        sample_covariance(np.zeros((1, 4)))  # need >= 2 observations


@pytest.mark.unit
def test_sample_covariance_accepts_1d_as_single_asset() -> None:
    """A 1-D return series is treated as a single-asset panel (1x1 covariance)."""
    cov = sample_covariance(np.array([0.01, -0.02, 0.03, 0.0]))
    assert cov.shape == (1, 1)
    assert cov[0, 0] > 0.0


@pytest.mark.unit
def test_markowitz_rejects_non_square_covariance() -> None:
    """A non-square covariance is rejected before any solve."""
    with pytest.raises(ValidationError):
        markowitz_weights(np.zeros((3, 2)))


@pytest.mark.unit
def test_markowitz_raises_on_singular_covariance() -> None:
    """A singular (rank-deficient, non-PD) covariance cannot be Cholesky-factored."""
    # Two identical assets => rank-1 singular covariance.
    cov = np.array([[1.0, 1.0], [1.0, 1.0]], dtype="float64")
    with pytest.raises(SingularCovarianceError):
        markowitz_weights(cov)


@pytest.mark.unit
def test_risk_parity_rejects_non_positive_diagonal() -> None:
    """risk-parity requires a strictly positive variance for every asset."""
    cov = np.array([[0.04, 0.0], [0.0, 0.0]], dtype="float64")
    with pytest.raises(ValidationError):
        risk_parity_weights(cov)


@pytest.mark.unit
def test_risk_parity_converges_on_correlated_covariance() -> None:
    """The damped fixed point converges (no iteration blow-up) on a correlated panel."""
    rng = np.random.default_rng(5)
    base = rng.normal(size=(600, 5))
    correlated = base + 0.4 * base[:, [0]]  # induce strong cross-correlation
    cov = sample_covariance(correlated)
    w = risk_parity_weights(cov)
    assert is_simplex(w)
    rc = w * (cov @ w)
    assert np.allclose(rc, rc.mean(), rtol=1e-2, atol=1e-6)


@pytest.mark.unit
def test_is_valid_baseline_weights_predicate() -> None:
    """The convenience predicate accepts a simplex and rejects a non-simplex."""
    assert is_valid_baseline_weights(equal_weight(4))
    assert not is_valid_baseline_weights(np.array([0.6, 0.6]))  # sums to 1.2
    assert not is_valid_baseline_weights(np.array([-0.5, 1.5]))  # negative entry


@pytest.mark.unit
def test_markowitz_rejects_non_finite_covariance() -> None:
    """A covariance with a non-finite entry is rejected at coercion."""
    cov = np.array([[0.04, np.inf], [np.inf, 0.09]], dtype="float64")
    with pytest.raises(ValidationError):
        markowitz_weights(cov)


@pytest.mark.unit
def test_risk_parity_returns_simplex_when_iteration_caps_out() -> None:
    """Even if the fixed point does not early-break, the returned weights are a valid simplex."""
    # A near-degenerate, highly-correlated covariance stresses the iteration so it
    # uses many steps; the result must still be a long-only simplex.
    n = 6
    base = 0.9 * np.ones((n, n)) + 0.1 * np.eye(n)
    cov = 0.02 * base
    w = risk_parity_weights(cov)
    assert is_simplex(w)
    assert bool((w >= 0.0).all())


@pytest.mark.unit
def test_baseline_result_to_dict_is_json_serializable(synthetic_panel: pd.DataFrame) -> None:
    """BaselineResult.to_dict yields plain JSON-friendly scalars and lists."""
    arr = synthetic_panel.to_numpy()
    result = run_baseline("equal_weight", arr[:400], arr[400:], cost_bps=5.0)
    assert isinstance(result, BaselineResult)
    payload = result.to_dict()
    assert payload["name"] == "equal_weight"
    assert isinstance(payload["turnover"], float)
    assert isinstance(payload["net_pnl"], float)
    assert isinstance(payload["n_bars"], int)
    assert isinstance(payload["weights"], list)
    assert isinstance(payload["equity_curve"], list)
    assert payload["meta"]["cost_bps"] == 5.0
