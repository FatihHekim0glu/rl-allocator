"""Additional coverage: polygon httpx path, dotenv, and remaining validation branches."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pytest

from rlallocator._exceptions import ValidationError
from rlallocator.agents.baselines import (
    baseline_weight_path,
    run_baseline,
    sample_covariance,
)
from rlallocator.data_providers.polygon import PolygonProvider, _load_api_key_from_dotenv
from rlallocator.env.backtester import equity_curve, vectorized_backtest
from rlallocator.evaluation.diebold_mariano import diebold_mariano
from rlallocator.evaluation.metrics import hac_standard_error, max_drawdown, net_pnl
from rlallocator.evaluation.verdict import derive_verdict


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self._i = 0

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get(self, url: str) -> _FakeResponse:
        resp = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return resp


@pytest.mark.unit
def test_polygon_httpx_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """The httpx path fetches and assembles a panel from a 200 payload."""
    import sys
    import types

    payload = {"status": "OK", "results": [{"t": 1_577_836_800_000, "c": 42.0}]}
    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = lambda *a, **k: _FakeClient([_FakeResponse(200, payload)])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    provider = PolygonProvider(api_key="K")
    frame = provider.fetch(["SPY"], date(2020, 1, 1), date(2020, 1, 2))
    assert float(frame["SPY"].iloc[0]) == 42.0


@pytest.mark.unit
def test_polygon_httpx_429_then_exhausts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated 429s exhaust the retry budget and raise RuntimeError (no sleep wait)."""
    import sys
    import types

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = lambda *a, **k: _FakeClient([_FakeResponse(429, {})])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    provider = PolygonProvider(api_key="K", max_retries=1, backoff_base=0.0)
    with pytest.raises(RuntimeError, match="429"):
        provider.fetch(["SPY"], date(2020, 1, 1), date(2020, 1, 2))


@pytest.mark.unit
def test_load_api_key_from_dotenv(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The dotenv reader parses POLYGON_API_KEY from a .env up the tree."""
    import rlallocator.data_providers.polygon as poly

    env_file = tmp_path / ".env"
    env_file.write_text("export POLYGON_API_KEY='DOTENVKEY'\n# comment\n", encoding="utf-8")
    fake_file = tmp_path / "src" / "rlallocator" / "data_providers" / "polygon.py"
    monkeypatch.setattr(poly, "__file__", str(fake_file))
    assert _load_api_key_from_dotenv() == "DOTENVKEY"


@pytest.mark.unit
def test_backtester_rejects_negative_cost() -> None:
    """The backtester rejects a negative cost_bps."""
    with pytest.raises(ValidationError):
        vectorized_backtest(np.zeros((3, 2)), np.zeros((3, 2)), cost_bps=-1.0)


@pytest.mark.unit
def test_backtester_initial_weights_shape_guard() -> None:
    """The backtester rejects initial_weights of the wrong length."""
    with pytest.raises(ValidationError):
        vectorized_backtest(np.zeros((3, 2)), np.zeros((3, 2)), initial_weights=np.zeros(3))


@pytest.mark.unit
def test_equity_curve_rejects_nonfinite() -> None:
    """equity_curve rejects an empty / non-finite series."""
    with pytest.raises(ValidationError):
        equity_curve(np.array([np.nan]))


@pytest.mark.unit
def test_baselines_reject_malformed_panels() -> None:
    """The baseline helpers reject malformed panels / asset mismatches."""
    with pytest.raises(ValidationError):
        sample_covariance(np.zeros((1, 3)))  # < 2 obs
    with pytest.raises(ValidationError):
        baseline_weight_path("equal_weight", np.zeros((10, 3)), n_oos_bars=0)
    with pytest.raises(ValidationError):
        run_baseline("equal_weight", np.zeros((10, 3)), np.zeros((10, 4)))


@pytest.mark.unit
def test_dm_constant_nonzero_differential_raises() -> None:
    """A constant non-zero differential makes the DM statistic undefined."""
    rl = np.full(20, 0.01)
    base = np.zeros(20)
    with pytest.raises(ValidationError):
        diebold_mariano(rl, base)


@pytest.mark.unit
def test_hac_standard_error_rejects_short_series() -> None:
    """The HAC SE needs at least two finite observations."""
    with pytest.raises(ValidationError):
        hac_standard_error(np.array([0.1]))


@pytest.mark.unit
def test_metrics_simple_edges() -> None:
    """max_drawdown / net_pnl handle a flat series cleanly."""
    assert max_drawdown(np.zeros(10)) == 0.0
    assert net_pnl(np.zeros(10)) == 0.0


@pytest.mark.unit
def test_verdict_rejects_nonfinite_inputs() -> None:
    """The verdict rejects non-finite evidence and a bad trial count."""
    base = {
        "dm_statistic": 1.0,
        "dm_pvalue": 0.1,
        "deflated_sharpe": 0.5,
        "seed_sharpe_lo": 0.1,
        "pbo": 0.3,
        "n_effective_trials": 5,
    }
    with pytest.raises(ValidationError):
        derive_verdict(**{**base, "dm_statistic": float("nan")})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        derive_verdict(**{**base, "seed_sharpe_lo": float("inf")})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        derive_verdict(**{**base, "n_effective_trials": 0})  # type: ignore[arg-type]
