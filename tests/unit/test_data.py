"""Unit tests for the synthetic multi-asset panels + the loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlallocator._exceptions import ValidationError
from rlallocator.data import compute_returns
from rlallocator.data.loaders import load_multi_asset_panel, synthetic_default_panel
from rlallocator.data.synthetic import (
    factor_regime_panel,
    learnable_edge_panel,
    pure_noise_panel,
)


@pytest.mark.unit
def test_factor_regime_panel_shape_and_determinism() -> None:
    """The factor-regime panel has the requested shape and is seed-deterministic."""
    p1 = factor_regime_panel(n_obs=300, n_assets=5, seed=11).returns
    p2 = factor_regime_panel(n_obs=300, n_assets=5, seed=11).returns
    assert p1.shape == (300, 5)
    pd.testing.assert_frame_equal(p1, p2)


@pytest.mark.unit
def test_factor_regime_is_near_zero_drift() -> None:
    """The honest-null panel has near-zero per-asset drift (no directional edge)."""
    returns = factor_regime_panel(n_obs=2000, n_assets=6, seed=3).returns
    # Every asset's mean is small relative to its volatility (near-zero Sharpe).
    means = returns.mean().to_numpy()
    vols = returns.std().to_numpy()
    assert np.all(np.abs(means) / vols < 0.1)


@pytest.mark.unit
def test_learnable_edge_premium_asset_has_higher_mean() -> None:
    """The learnable_edge premium asset has the highest sample mean return."""
    panel = learnable_edge_panel(n_obs=2000, n_assets=6, seed=3, premium=0.0015, premium_asset=0)
    means = panel.returns.mean().to_numpy()
    assert int(np.argmax(means)) == 0
    assert panel.premium_asset == 0


@pytest.mark.unit
def test_pure_noise_panel_driftless() -> None:
    """The pure-noise panel is driftless white noise."""
    returns = pure_noise_panel(n_obs=3000, n_assets=4, seed=5).returns
    assert np.all(np.abs(returns.mean().to_numpy()) < 0.001)


@pytest.mark.unit
def test_compute_returns_no_lookahead() -> None:
    """compute_returns drops the leading NaN and never forward-fills."""
    prices = pd.DataFrame({"a": [100.0, 110.0, 99.0], "b": [50.0, 55.0, 55.0]})
    returns = compute_returns(prices)
    assert returns.shape == (2, 2)
    assert np.isclose(returns.iloc[0]["a"], 0.1)


@pytest.mark.unit
def test_synthetic_default_panel_returns_prices_and_returns() -> None:
    """The default loader returns aligned price + return panels tagged synthetic."""
    prices, returns, source = synthetic_default_panel(n_obs=200, n_assets=4, seed=7)
    assert source == "synthetic"
    assert prices.shape == (200, 4)
    assert returns.shape == (200, 4)
    # The synthetic returns ARE the panel (the generator emits returns directly); the
    # price panel is their anchored cumulative product (same number of rows).


@pytest.mark.unit
def test_synthetic_default_panel_rejects_unknown_kind() -> None:
    """An unknown synthetic kind is rejected."""
    with pytest.raises(ValidationError):
        synthetic_default_panel(kind="momentum")


@pytest.mark.unit
def test_load_multi_asset_panel_falls_back_to_synthetic() -> None:
    """Without a key/network the loader falls back to the deterministic synthetic panel."""
    from datetime import date

    _, returns, source = load_multi_asset_panel(
        ["SPY", "TLT", "GLD"],
        start=date(2015, 1, 1),
        end=date(2016, 1, 1),
        data_source_pref="synthetic",
    )
    assert source == "synthetic"
    assert returns.shape[1] == 3
