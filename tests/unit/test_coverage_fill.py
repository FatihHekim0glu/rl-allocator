"""Unit tests filling remaining error-branch / lazy-path coverage gaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlallocator._exceptions import ValidationError
from rlallocator._validation import ensure_series, is_simplex
from rlallocator.evaluation.metrics import oos_sharpe, turnover
from rlallocator.evaluation.seed_lottery import seed_lottery, variance_of_seed_sharpes
from rlallocator.plots import equity_curve_figure, seed_lottery_figure, weights_area_figure


@pytest.mark.unit
def test_seed_lottery_single_seed_falls_back_to_empirical() -> None:
    """A single-seed lottery collapses both bounds onto that seed's Sharpe."""
    result = seed_lottery(np.array([0.3]), seed=7)
    assert result.n_seeds == 1
    assert result.sharpe_lo == result.sharpe_hi == result.median_sharpe == 0.3
    assert result.sharpe_std == 0.0


@pytest.mark.unit
def test_seed_lottery_bootstrap_multi_seed() -> None:
    """A multi-seed lottery bootstraps a band that brackets the median."""
    sharpes = np.array([-0.2, 0.0, 0.1, 0.3, 0.5, -0.1])
    result = seed_lottery(sharpes, seed=11)
    assert result.sharpe_lo <= result.median_sharpe <= result.sharpe_hi
    assert result.sharpe_std > 0.0
    assert len(result.to_dict()["seed_sharpes"]) == 6


@pytest.mark.unit
def test_seed_lottery_rejects_bad_args() -> None:
    """seed_lottery rejects an empty / non-finite set and an out-of-range alpha."""
    with pytest.raises(ValidationError):
        seed_lottery(np.array([]))
    with pytest.raises(ValidationError):
        seed_lottery(np.array([np.nan, 0.1]))
    with pytest.raises(ValidationError):
        seed_lottery(np.array([0.1, 0.2]), alpha=1.5)


@pytest.mark.unit
def test_variance_of_seed_sharpes_needs_two() -> None:
    """The cross-seed variance needs at least two seeds."""
    with pytest.raises(ValidationError):
        variance_of_seed_sharpes(np.array([0.3]))


@pytest.mark.unit
def test_plots_reject_empty_and_mismatched() -> None:
    """The figure builders reject empty / non-finite / one-edge-only inputs."""
    with pytest.raises(ValidationError):
        weights_area_figure(np.empty((0, 3)))
    with pytest.raises(ValidationError):
        seed_lottery_figure(np.array([np.inf]), best_baseline_sharpe=0.1)
    with pytest.raises(ValidationError):
        equity_curve_figure(
            rl_median_equity=np.array([1.0, 1.1]),
            baseline_equities={"equal_weight": [1.0, 1.05]},
            seed_band_lo=[1.0, 0.9],  # only one band edge
        )


@pytest.mark.unit
def test_equity_curve_figure_with_band() -> None:
    """The equity figure renders the across-seed band (two extra traces)."""
    rl = np.array([1.0, 1.1, 1.2])
    fig = equity_curve_figure(
        rl_median_equity=rl,
        baseline_equities={"equal_weight": [1.0, 1.02, 1.04]},
        seed_band_lo=[0.95, 1.0, 1.05],
        seed_band_hi=[1.05, 1.15, 1.25],
    )
    # 2 band traces + RL + 1 baseline = 4
    assert len(fig["data"]) == 4


@pytest.mark.unit
def test_metrics_reject_nonfinite() -> None:
    """The scalar metrics reject non-finite / empty series."""
    with pytest.raises(ValidationError):
        oos_sharpe(np.array([np.nan, 0.1]))
    with pytest.raises(ValidationError):
        turnover(np.empty((0, 3)))


@pytest.mark.unit
def test_ensure_series_and_simplex_edges() -> None:
    """ensure_series rejects 2-D input; is_simplex rejects empty / non-finite."""
    with pytest.raises(ValidationError):
        ensure_series(np.zeros((2, 2)))
    assert not is_simplex([])
    assert not is_simplex([np.nan, 1.0])


@pytest.mark.unit
def test_serve_figures_with_injected_committed_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When committed metrics exist, run_allocation emits both figures + the RL summary."""
    import rlallocator.serve as serve_mod

    # A minimal committed-metrics payload exercising the figure + summary branches.
    payload = {
        "oos_sharpe_rl_median": 0.12,
        "seed_sharpe_lo": -0.05,
        "seed_sharpe_hi": 0.30,
        "dm_pvalue_vs_best": 0.4,
        "deflated_sharpe": 0.6,
        "pbo": 0.55,
        "turnover": 3.2,
        "max_drawdown": -0.1,
        "rl_beats_baselines": False,
        "n_effective_trials": 5,
        "rl_median_equity": list(np.cumprod(1.0 + np.full(40, 0.001))),
        "seed_band_lo": list(np.cumprod(1.0 + np.full(40, 0.0005))),
        "seed_band_hi": list(np.cumprod(1.0 + np.full(40, 0.0015))),
        "rl_weight_path": [[0.5, 0.25, 0.25]] * 40,
    }
    monkeypatch.setattr(serve_mod, "_read_committed_metrics", lambda: payload)
    run = serve_mod.run_allocation(n_assets=3, n_seeds=5, cost_bps=10.0, lookback=8, seed=7)
    assert run.summary.oos_sharpe_rl_median == 0.12
    assert run.equity_figure  # non-empty (committed RL equity present)
    assert run.weights_figure  # non-empty (committed weight path present)


@pytest.mark.unit
def test_serve_read_committed_metrics_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_read_committed_metrics returns {} when the artifact is absent or corrupt."""
    import rlallocator.serve as serve_mod

    monkeypatch.setattr(serve_mod, "_ARTIFACTS_DIR", tmp_path)
    assert serve_mod._read_committed_metrics() == {}
    (tmp_path / "metrics.json").write_text("not json{", encoding="utf-8")
    assert serve_mod._read_committed_metrics() == {}
    (tmp_path / "metrics.json").write_text(json.dumps({"pbo": 0.3}), encoding="utf-8")
    assert serve_mod._read_committed_metrics() == {"pbo": 0.3}
