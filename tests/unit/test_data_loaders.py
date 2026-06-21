"""Loader tests for the data group: synthetic default + the offline PIT fallback.

Covers the loader contract the deployed serve path depends on:

- the synthetic default is deterministic and emits aligned price / return panels with
  the ``"synthetic"`` provenance, for every panel kind;
- ``compute_returns`` honours the NO-LOOKAHEAD rule (``pct_change(fill_method=None)``):
  a price gap (a repeated level) yields an honest zero return, never a forward-filled
  manufactured one, and an actual ``NaN`` price propagates rather than being filled;
- ``load_multi_asset_panel`` falls back to the deterministic synthetic panel offline
  (no key / no network / ``data`` extra absent) and tags provenance correctly;
- the business-day span drives the synthetic fallback length.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from rlallocator._exceptions import ValidationError
from rlallocator.data import compute_returns
from rlallocator.data.loaders import (
    load_multi_asset_panel,
    synthetic_default_panel,
)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["factor_regime", "learnable_edge", "pure_noise"])
def test_synthetic_default_panel_is_deterministic_for_each_kind(kind: str) -> None:
    """Each synthetic panel kind reproduces price + return panels under a fixed seed."""
    p1, r1, s1 = synthetic_default_panel(n_obs=120, n_assets=4, seed=9, kind=kind)
    p2, r2, s2 = synthetic_default_panel(n_obs=120, n_assets=4, seed=9, kind=kind)
    assert s1 == s2 == "synthetic"
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(r1, r2)
    assert p1.shape == (120, 4)


@pytest.mark.unit
def test_synthetic_default_prices_are_anchored_and_consistent() -> None:
    """The synthetic price panel is the anchored cumulative product of (1 + returns)."""
    prices, returns, _ = synthetic_default_panel(n_obs=200, n_assets=3, seed=4)
    expected = 100.0 * (1.0 + returns).cumprod()
    pd.testing.assert_frame_equal(prices, expected.astype("float64"))
    assert np.all(prices.to_numpy() > 0.0)


@pytest.mark.unit
def test_compute_returns_no_forward_fill_on_repeated_level() -> None:
    """A repeated price level yields an honest zero return (not a filled-over gap)."""
    prices = pd.DataFrame({"a": [100.0, 100.0, 102.0], "b": [10.0, 11.0, 11.0]})
    returns = compute_returns(prices)
    assert returns.shape == (2, 2)
    # Repeated level -> exactly zero return; this is NOT a manufactured ffill artifact.
    assert returns.iloc[0]["a"] == pytest.approx(0.0)
    assert returns.iloc[1]["b"] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_returns_propagates_nan_gap() -> None:
    """An actual NaN price propagates (never forward-filled before differencing)."""
    prices = pd.DataFrame({"a": [100.0, np.nan, 110.0]})
    returns = compute_returns(prices)
    # With fill_method=None the NaN row produces NaN returns rather than a fake 0/value.
    assert bool(returns["a"].isna().any())


@pytest.mark.unit
def test_compute_returns_rejects_too_short_panel() -> None:
    """A single-row price panel cannot form a return and is rejected."""
    with pytest.raises(ValidationError):
        compute_returns(pd.DataFrame({"a": [100.0]}))


@pytest.mark.unit
def test_load_multi_asset_panel_synthetic_is_deterministic() -> None:
    """The offline fallback path is deterministic for a fixed (tickers, dates, seed)."""
    args = dict(start=date(2015, 1, 1), end=date(2016, 1, 1), data_source_pref="synthetic")
    _, r1, s1 = load_multi_asset_panel(["SPY", "TLT", "GLD"], seed=7, **args)  # type: ignore[arg-type]
    _, r2, s2 = load_multi_asset_panel(["SPY", "TLT", "GLD"], seed=7, **args)  # type: ignore[arg-type]
    assert s1 == s2 == "synthetic"
    pd.testing.assert_frame_equal(r1, r2)
    assert r1.shape[1] == 3


@pytest.mark.unit
def test_load_multi_asset_panel_span_drives_length() -> None:
    """A longer business-day span yields a longer synthetic fallback panel."""
    short = load_multi_asset_panel(
        ["A", "B"], start=date(2020, 1, 1), end=date(2020, 2, 1), data_source_pref="synthetic"
    )[1]
    long = load_multi_asset_panel(
        ["A", "B"], start=date(2020, 1, 1), end=date(2020, 12, 31), data_source_pref="synthetic"
    )[1]
    assert long.shape[0] > short.shape[0]


@pytest.mark.unit
def test_load_multi_asset_panel_polygon_falls_back_without_key() -> None:
    """The polygon path falls back to synthetic when the provider cannot fetch (no key)."""
    _, returns, source = load_multi_asset_panel(
        ["SPY", "TLT"],
        start=date(2019, 1, 1),
        end=date(2020, 1, 1),
        data_source_pref="polygon",
    )
    # Offline / no key: the provider fetch fails and the loader returns synthetic.
    assert source == "synthetic"
    assert returns.shape[1] == 2


@pytest.mark.unit
def test_load_multi_asset_panel_polygon_success_tags_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful polygon fetch yields the real panel, ``polygon`` provenance, and PIT returns."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    fake_prices = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0, 101.0, 103.0], "TLT": [50.0, 50.5, 50.0, 49.5, 50.0]},
        index=idx,
    )

    class _FakeProvider:
        def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
            return fake_prices[tickers]

    import rlallocator.data_providers.polygon as poly

    monkeypatch.setattr(poly, "PolygonProvider", _FakeProvider)
    prices, returns, source = load_multi_asset_panel(
        ["SPY", "TLT"],
        start=date(2020, 1, 1),
        end=date(2020, 1, 8),
        data_source_pref="polygon",
    )
    assert source == "polygon"
    assert list(prices.columns) == ["SPY", "TLT"]
    # Returns are PIT (pct_change(fill_method=None)) with the leading NaN row dropped.
    assert returns.shape == (4, 2)
    assert returns.iloc[0]["SPY"] == pytest.approx(0.01)


@pytest.mark.unit
def test_load_multi_asset_panel_polygon_empty_payload_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty provider payload falls back to the deterministic synthetic panel."""

    class _EmptyProvider:
        def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
            return pd.DataFrame()

    import rlallocator.data_providers.polygon as poly

    monkeypatch.setattr(poly, "PolygonProvider", _EmptyProvider)
    _, returns, source = load_multi_asset_panel(
        ["SPY", "TLT"],
        start=date(2020, 1, 1),
        end=date(2020, 2, 1),
        data_source_pref="polygon",
    )
    assert source == "synthetic"
    assert returns.shape[1] == 2
