"""Structure tests for the synthetic multi-asset DGPs (the data group's deliverable).

These pin the *statistical* contract the rest of the honest-null stack relies on,
mirroring the brief's data test plan:

- **determinism** — a master seed reproduces a panel byte-for-byte, and distinct
  seeds give distinct panels (the substream design is seed-driven, not global state);
- **factor / regime correlation structure** — the factor model induces a strictly
  POSITIVE cross-asset correlation (a common market factor co-moves the basket) and
  the regimes scale factor volatility (regime-switching vol/correlation), while
  carrying no directional edge (the honest NULL: no asset has a tradeable Sharpe);
- **learnable_edge has a TRADEABLE premium** — the premium asset has the highest
  risk-adjusted return (Sharpe), and a static tilt toward it BEATS 1/N risk-adjusted
  (so an allocator that works is not chasing a vacuous null);
- **pure_noise has NONE** — driftless white noise with no cross-asset correlation and
  no per-asset Sharpe edge (the strict null);
- **validation** — the shared dimension / volatility / premium-index preconditions
  reject bad inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlallocator._exceptions import ValidationError
from rlallocator.data.synthetic import (
    ReturnPanelData,
    factor_regime_panel,
    learnable_edge_panel,
    pure_noise_panel,
)


def _sharpe(returns: np.ndarray) -> np.ndarray:
    """Per-column (per-asset or per-strategy) sample Sharpe of a return array."""
    arr = np.asarray(returns, dtype="float64")
    sharpe: np.ndarray = arr.mean(axis=0) / arr.std(axis=0)
    return sharpe


# --------------------------------------------------------------------------- #
# determinism                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_factor_regime_seed_determinism_and_separation() -> None:
    """Identical seeds reproduce a panel exactly; distinct seeds give distinct panels."""
    a = factor_regime_panel(n_obs=400, n_assets=5, seed=11)
    b = factor_regime_panel(n_obs=400, n_assets=5, seed=11)
    c = factor_regime_panel(n_obs=400, n_assets=5, seed=12)
    pd.testing.assert_frame_equal(a.returns, b.returns)
    assert a.regime_labels == b.regime_labels
    # A different master seed must move the panel (the DGP is seed-driven).
    assert not np.allclose(a.returns.to_numpy(), c.returns.to_numpy())


@pytest.mark.unit
def test_all_three_dgps_are_deterministic() -> None:
    """Every synthetic DGP reproduces byte-for-byte under a fixed seed."""
    pd.testing.assert_frame_equal(
        learnable_edge_panel(n_obs=300, n_assets=4, seed=5, premium=0.001).returns,
        learnable_edge_panel(n_obs=300, n_assets=4, seed=5, premium=0.001).returns,
    )
    pd.testing.assert_frame_equal(
        pure_noise_panel(n_obs=300, n_assets=4, seed=5).returns,
        pure_noise_panel(n_obs=300, n_assets=4, seed=5).returns,
    )


@pytest.mark.unit
def test_panel_frame_index_and_columns() -> None:
    """The panel is a business-day-indexed, ``asset_i``-columned float64 frame."""
    panel = factor_regime_panel(n_obs=50, n_assets=3, seed=1)
    frame = panel.returns
    assert list(frame.columns) == ["asset_0", "asset_1", "asset_2"]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.to_numpy().dtype == np.float64
    assert len(panel.regime_labels) == 50


# --------------------------------------------------------------------------- #
# factor / regime correlation structure                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_factor_model_induces_positive_cross_asset_correlation() -> None:
    """The common market factor co-moves the basket: all pairwise correlations > 0."""
    returns = factor_regime_panel(n_obs=4000, n_assets=6, n_factors=3, seed=3).returns
    corr = np.corrcoef(returns.to_numpy(), rowvar=False)
    off_diag = corr[np.triu_indices(corr.shape[0], k=1)]
    # A shared positive-loading market factor makes every pair POSITIVELY correlated.
    assert float(off_diag.min()) > 0.05
    assert float(off_diag.mean()) > 0.20


@pytest.mark.unit
def test_regimes_scale_volatility() -> None:
    """The regime label drives factor volatility: regimes have distinct vol levels."""
    panel = factor_regime_panel(n_obs=6000, n_assets=6, n_regimes=2, seed=3)
    returns = panel.returns.to_numpy()
    labels = np.asarray(panel.regime_labels)
    assert set(np.unique(labels)) == {0, 1}
    vols = {int(r): float(returns[labels == r].std(axis=0).mean()) for r in np.unique(labels)}
    # The two regimes scale the common factor volatility differently (vol-switching).
    lo, hi = sorted(vols.values())
    assert hi > 1.2 * lo


@pytest.mark.unit
def test_factor_regime_is_an_honest_null_no_tradeable_asset() -> None:
    """No asset in the null panel carries a tradeable Sharpe edge (no directional edge)."""
    returns = factor_regime_panel(n_obs=4000, n_assets=6, seed=3).returns.to_numpy()
    per_asset_sharpe = _sharpe(returns)
    # Every asset's per-bar Sharpe is tiny: nothing to tilt toward.
    assert float(np.abs(per_asset_sharpe).max()) < 0.10


@pytest.mark.unit
def test_single_regime_yields_constant_labels() -> None:
    """With ``n_regimes == 1`` the latent-regime labels are all zero (no switching)."""
    panel = factor_regime_panel(n_obs=200, n_assets=4, n_regimes=1, seed=2)
    assert set(panel.regime_labels) == {0}


@pytest.mark.unit
def test_single_factor_still_co_moves_assets() -> None:
    """A single market factor (``n_factors == 1``) still positively co-moves the basket."""
    returns = factor_regime_panel(n_obs=4000, n_assets=5, n_factors=1, seed=3).returns
    corr = np.corrcoef(returns.to_numpy(), rowvar=False)
    off_diag = corr[np.triu_indices(corr.shape[0], k=1)]
    assert float(off_diag.min()) > 0.0


# --------------------------------------------------------------------------- #
# learnable_edge has a TRADEABLE premium                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_learnable_edge_premium_asset_has_highest_sharpe() -> None:
    """The premium asset has the single highest risk-adjusted return (Sharpe)."""
    panel = learnable_edge_panel(n_obs=4000, n_assets=6, seed=3, premium=0.0015, premium_asset=2)
    sharpes = _sharpe(panel.returns.to_numpy())
    assert int(np.argmax(sharpes)) == 2
    assert panel.premium_asset == 2


@pytest.mark.unit
def test_learnable_edge_tilt_beats_equal_weight_risk_adjusted() -> None:
    """A static tilt toward the premium asset BEATS 1/N risk-adjusted (tradeable edge)."""
    returns = learnable_edge_panel(
        n_obs=4000, n_assets=6, seed=3, premium=0.0015, premium_asset=0
    ).returns.to_numpy()
    n_assets = returns.shape[1]
    equal_w = np.full(n_assets, 1.0 / n_assets)
    tilt_w = np.full(n_assets, 0.05)
    tilt_w[0] = 1.0 - 0.05 * (n_assets - 1)
    eq_sharpe = float(_sharpe(returns @ equal_w))
    tilt_sharpe = float(_sharpe(returns @ tilt_w))
    # The premium is genuinely TRADEABLE: concentrating into it lifts the Sharpe.
    assert tilt_sharpe > eq_sharpe


@pytest.mark.unit
def test_factor_regime_tilt_does_not_beat_equal_weight() -> None:
    """On the honest null the same tilt does NOT beat 1/N (the edge is absent)."""
    returns = factor_regime_panel(n_obs=4000, n_assets=6, seed=3).returns.to_numpy()
    n_assets = returns.shape[1]
    equal_w = np.full(n_assets, 1.0 / n_assets)
    tilt_w = np.full(n_assets, 0.05)
    tilt_w[0] = 1.0 - 0.05 * (n_assets - 1)
    eq_sharpe = float(_sharpe(returns @ equal_w))
    tilt_sharpe = float(_sharpe(returns @ tilt_w))
    # No tradeable edge: a concentrated tilt is NOT rewarded (honest, not vacuous null).
    assert tilt_sharpe <= eq_sharpe


# --------------------------------------------------------------------------- #
# pure_noise has NONE                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_pure_noise_has_no_cross_asset_correlation() -> None:
    """The strict null has ~zero cross-asset correlation (no common factor)."""
    returns = pure_noise_panel(n_obs=5000, n_assets=6, seed=3).returns.to_numpy()
    corr = np.corrcoef(returns, rowvar=False)
    off_diag = corr[np.triu_indices(corr.shape[0], k=1)]
    assert float(np.abs(off_diag).max()) < 0.10


@pytest.mark.unit
def test_pure_noise_has_no_per_asset_sharpe_edge() -> None:
    """The strict null carries no tradeable per-asset Sharpe (driftless white noise)."""
    returns = pure_noise_panel(n_obs=5000, n_assets=6, seed=3).returns.to_numpy()
    assert float(np.abs(_sharpe(returns)).max()) < 0.06


@pytest.mark.unit
def test_pure_noise_single_regime_label() -> None:
    """The pure-noise panel reports a single nominal regime label per bar."""
    panel = pure_noise_panel(n_obs=120, n_assets=3, seed=1)
    assert set(panel.regime_labels) == {0}
    assert panel.premium_asset == -1
    assert panel.kind == "pure_noise"


# --------------------------------------------------------------------------- #
# ReturnPanelData metadata                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_return_panel_to_dict_omits_matrix() -> None:
    """``to_dict`` emits JSON-safe shape metadata only (not the full return matrix)."""
    panel = learnable_edge_panel(n_obs=40, n_assets=4, seed=1, premium_asset=1)
    meta = panel.to_dict()
    assert meta["n_obs"] == 40
    assert meta["n_assets"] == 4
    assert meta["kind"] == "learnable_edge"
    assert meta["premium_asset"] == 1
    assert len(meta["regime_labels"]) == 40
    assert "returns" not in meta
    assert isinstance(panel, ReturnPanelData)


# --------------------------------------------------------------------------- #
# validation                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_obs": 1},
        {"n_assets": 1},
        {"n_factors": 0},
        {"n_regimes": 0},
        {"base_vol": -0.01},
        {"idio_vol": float("nan")},
    ],
)
def test_factor_regime_rejects_bad_args(kwargs: dict[str, float]) -> None:
    """Every shared dimension / volatility precondition is enforced."""
    with pytest.raises(ValidationError):
        factor_regime_panel(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("bad_index", [-1, 6])
def test_learnable_edge_rejects_out_of_range_premium_asset(bad_index: int) -> None:
    """``premium_asset`` outside ``[0, n_assets)`` is rejected."""
    with pytest.raises(ValidationError):
        learnable_edge_panel(n_obs=50, n_assets=6, premium_asset=bad_index)


@pytest.mark.unit
def test_pure_noise_rejects_negative_vol() -> None:
    """A negative per-bar volatility is rejected by the pure-noise DGP."""
    with pytest.raises(ValidationError):
        pure_noise_panel(n_obs=50, n_assets=4, vol=-1.0)
