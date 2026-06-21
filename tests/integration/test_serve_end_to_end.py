"""Integration: end-to-end synthetic -> walk-forward baselines -> verdict (no torch/sb3)."""

from __future__ import annotations

import sys

import pytest

from rlallocator.serve import RlAllocatorRun, run_allocation


@pytest.mark.integration
def test_end_to_end_run_allocation_synthetic() -> None:
    """run_allocation produces a JSON-safe run on the synthetic panel, torch-free."""
    run = run_allocation(
        n_assets=6, n_seeds=5, cost_bps=10.0, lookback=32, rebalance="monthly", seed=7
    )
    assert isinstance(run, RlAllocatorRun)
    payload = run.to_dict()

    summary = payload["summary"]
    for key in (
        "oos_sharpe_rl_median",
        "oos_sharpe_1n",
        "oos_sharpe_markowitz",
        "oos_sharpe_riskparity",
        "best_baseline",
        "dm_pvalue_vs_best",
        "deflated_sharpe",
        "pbo",
        "rl_beats_baselines",
        "n_effective_trials",
        "data_source",
    ):
        assert key in summary, key

    # The honest-NULL default: with no committed policy the verdict is False.
    assert summary["rl_beats_baselines"] is False
    assert summary["data_source"] == "synthetic"

    # No torch / sb3 / gymnasium was imported by the serve path.
    assert "torch" not in sys.modules
    assert "stable_baselines3" not in sys.modules
    assert "gymnasium" not in sys.modules


@pytest.mark.integration
def test_end_to_end_is_json_serializable() -> None:
    """The run payload is fully JSON-serializable (no numpy scalars / Plotly objects)."""
    import json

    run = run_allocation(n_assets=4, n_seeds=3, cost_bps=10.0, lookback=16, seed=11)
    text = json.dumps(run.to_dict())
    assert isinstance(text, str)
    assert len(text) > 0
