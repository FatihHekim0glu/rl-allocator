"""Unit tests for the vendored Polygon provider (key resolution + payload parsing + retries)."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd
import pytest

from rlallocator._exceptions import ValidationError
from rlallocator.data_providers.polygon import PolygonProvider, _resolve_api_key


@pytest.mark.unit
def test_resolve_api_key_explicit() -> None:
    """An explicit key takes precedence."""
    assert _resolve_api_key("EXPLICIT") == "EXPLICIT"


@pytest.mark.unit
def test_resolve_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key resolves from the POLYGON_API_KEY environment variable."""
    monkeypatch.setenv("POLYGON_API_KEY", "ENVKEY")
    assert _resolve_api_key(None) == "ENVKEY"


@pytest.mark.unit
def test_resolve_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent any source the key resolution raises ValidationError."""
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(
        "rlallocator.data_providers.polygon._load_api_key_from_dotenv", lambda: None
    )
    with pytest.raises(ValidationError):
        _resolve_api_key(None)


@pytest.mark.unit
def test_fetch_validates_inputs() -> None:
    """fetch rejects empty tickers and a non-increasing date range."""
    provider = PolygonProvider(api_key="K")
    with pytest.raises(ValidationError):
        provider.fetch([], date(2020, 1, 1), date(2020, 2, 1))
    with pytest.raises(ValidationError):
        provider.fetch(["SPY"], date(2020, 2, 1), date(2020, 1, 1))


@pytest.mark.unit
def test_series_from_payload_parses_bars() -> None:
    """A Polygon aggregates payload parses into a date-indexed close series."""
    payload: dict[str, Any] = {
        "status": "OK",
        "results": [
            {"t": 1_577_836_800_000, "c": 100.0},
            {"t": 1_577_923_200_000, "c": 101.5},
        ],
    }
    series = PolygonProvider._series_from_payload(payload, "SPY")
    assert list(series.values) == [100.0, 101.5]
    assert series.name == "SPY"


@pytest.mark.unit
def test_series_from_payload_empty_raises() -> None:
    """An empty results payload raises."""
    with pytest.raises(ValueError, match="no results"):
        PolygonProvider._series_from_payload({"status": "OK", "results": []}, "SPY")


@pytest.mark.unit
def test_fetch_uses_urllib_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """With httpx absent, fetch parses via the urllib fallback and builds the panel."""
    provider = PolygonProvider(api_key="K")
    payload = json.dumps({"status": "OK", "results": [{"t": 1_577_836_800_000, "c": 50.0}]})

    def _fake_urllib(url: str) -> dict[str, Any]:
        return json.loads(payload)

    monkeypatch.setattr(provider, "_get_json_urllib", _fake_urllib)

    # Force the httpx-absent branch by making the lazy import fail.
    import builtins

    real_import = builtins.__import__

    def _no_httpx(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_httpx)
    frame = provider.fetch(["SPY"], date(2020, 1, 1), date(2020, 1, 2))
    assert isinstance(frame, pd.DataFrame)
    assert "SPY" in frame.columns
    assert float(frame["SPY"].iloc[0]) == 50.0
