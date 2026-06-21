"""Regression: the served path computes OOS metrics via the PURGED walk-forward.

The #1 recurring bug is "the entrypoint bypasses the rigorous fn" — the served
``run_allocation`` quietly scoring the FULL sample instead of the concatenated purged
walk-forward folds. This test pins the wiring two ways:

1. a SOURCE grep asserting ``serve.py`` imports and calls ``make_folds`` (the
   walk-forward fn) — the static guard;
2. a BEHAVIORAL check asserting ``run_allocation`` actually invokes ``make_folds`` at
   runtime (a spy) — the dynamic guard.
"""

from __future__ import annotations

import inspect

import pytest

import rlallocator.serve as serve_mod
from rlallocator.serve import run_allocation


@pytest.mark.regression
def test_serve_source_calls_make_folds() -> None:
    """The serve module source imports and calls the walk-forward fn make_folds."""
    source = inspect.getsource(serve_mod)
    assert "make_folds" in source, "serve.py must use the purged walk-forward (make_folds)."
    # And it must be CALLED, not merely imported.
    assert "make_folds(" in source


@pytest.mark.regression
def test_run_allocation_invokes_walk_forward_at_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_allocation calls make_folds at runtime (the served OOS metrics are walk-forward)."""
    calls: list[int] = []
    real_make_folds = serve_mod.make_folds

    def _spy(n_bars: int, **kwargs: object) -> object:
        calls.append(n_bars)
        return real_make_folds(n_bars, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(serve_mod, "make_folds", _spy)
    run_allocation(n_assets=4, n_seeds=3, cost_bps=10.0, lookback=16, seed=7)
    assert calls, "run_allocation must call make_folds (the purged walk-forward)."


@pytest.mark.regression
def test_walk_forward_oos_differs_from_full_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """The walk-forward OOS Sharpes are not a trivial full-sample fit (folds are used)."""
    n_folds_seen: list[int] = []
    real_make_folds = serve_mod.make_folds

    def _spy(n_bars: int, **kwargs: object) -> object:
        folds = real_make_folds(n_bars, **kwargs)  # type: ignore[arg-type]
        n_folds_seen.append(len(folds))
        return folds

    monkeypatch.setattr(serve_mod, "make_folds", _spy)
    run_allocation(n_assets=4, n_seeds=3, cost_bps=10.0, lookback=16, seed=7)
    # The served path splits into multiple OOS folds (not one full-sample window).
    assert n_folds_seen and n_folds_seen[0] >= 2
