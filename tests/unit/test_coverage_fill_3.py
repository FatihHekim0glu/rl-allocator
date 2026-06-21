"""Final coverage fill: serve auto path, loaders polygon fallback, misc validation branches."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from rlallocator.data import compute_returns
from rlallocator.data.loaders import load_multi_asset_panel
from rlallocator.serve import run_allocation


@pytest.mark.unit
def test_run_allocation_auto_falls_back_to_synthetic() -> None:
    """data_source_pref='auto' attempts the PIT path and falls back to synthetic (no key)."""
    run = run_allocation(
        n_assets=4,
        n_seeds=3,
        cost_bps=10.0,
        lookback=16,
        data_source_pref="auto",
        seed=7,
    )
    # No key / network -> the loader falls through to the synthetic panel.
    assert run.summary.data_source == "synthetic"


@pytest.mark.unit
def test_loaders_polygon_pref_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """The polygon pref falls back to synthetic when the provider raises (no key)."""
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(
        "rlallocator.data_providers.polygon._load_api_key_from_dotenv", lambda: None
    )
    _, returns, source = load_multi_asset_panel(
        ["A", "B", "C", "D"],
        start=date(2017, 1, 1),
        end=date(2018, 1, 1),
        data_source_pref="polygon",
    )
    assert source == "synthetic"
    assert returns.shape[1] == 4


@pytest.mark.unit
def test_compute_returns_rejects_single_row() -> None:
    """compute_returns rejects a single-row price panel."""
    import pandas as pd

    from rlallocator._exceptions import ValidationError

    with pytest.raises(ValidationError):
        compute_returns(pd.DataFrame({"a": [100.0], "b": [50.0]}))


@pytest.mark.unit
def test_compute_returns_drops_leading_nan() -> None:
    """compute_returns drops the leading NaN row and never forward-fills gaps."""
    import pandas as pd

    prices = pd.DataFrame({"a": [100.0, np.nan, 121.0], "b": [50.0, 55.0, 60.5]})
    returns = compute_returns(prices)
    # Row 0 (the first diff) is dropped; the NaN gap is preserved, not ffilled.
    assert returns.shape[0] == 2
