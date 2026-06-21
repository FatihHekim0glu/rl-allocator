"""Shared, seeded test fixtures (multi-asset return panels).

Every fixture is deterministic (driven by :func:`rlallocator._rng.make_rng` via the
synthetic generators) and returns a multi-asset return PANEL with known structure, so
tests across the suite share identical synthetic data:

- ``synthetic_panel`` — the deployed-default factor + regime-switching panel (the
  honest NULL: no allocation beats 1/N net of costs);
- ``learnable_edge`` — the same factor structure but one asset carries a persistent
  risk-adjusted PREMIUM (the SANITY fixture: an allocator that works SHOULD tilt toward
  it and beat 1/N — so the null is honest, not vacuous);
- ``pure_noise`` — driftless i.i.d. cross-sectional white noise (the strict null).

Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlallocator._rng import make_rng
from rlallocator.data.synthetic import (
    factor_regime_panel,
    learnable_edge_panel,
    pure_noise_panel,
)

_SEED = 20260621
_N_OBS = 768
_N_ASSETS = 6
#: The index of the premium-bearing asset in the ``learnable_edge`` fixture.
LEARNABLE_EDGE_PREMIUM_ASSET = 0


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(_SEED)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """Deployed-default factor + regime-switching multi-asset return panel (the honest NULL).

    Every asset shares the same (near-zero) unconditional risk-adjusted return, so no
    static or dynamic re-weighting reliably beats 1/N net of costs. Shape
    ``(768, 6)``, seeded for byte-identical reproduction.
    """
    return factor_regime_panel(n_obs=_N_OBS, n_assets=_N_ASSETS, seed=_SEED).returns


@pytest.fixture
def learnable_edge() -> pd.DataFrame:
    """A factor panel where ONE asset carries a persistent premium (the SANITY fixture).

    Asset ``0`` has a constant positive excess drift on top of its factor return — a
    higher Sharpe than its peers. An allocator whose env + training work SHOULD tilt
    toward it and beat 1/N (proving the machinery works, so the null is honest, not
    vacuous). Shape ``(768, 6)``, seeded.
    """
    return learnable_edge_panel(
        n_obs=_N_OBS,
        n_assets=_N_ASSETS,
        seed=_SEED,
        premium=0.0015,
        premium_asset=LEARNABLE_EDGE_PREMIUM_ASSET,
    ).returns


@pytest.fixture
def pure_noise() -> pd.DataFrame:
    """A driftless i.i.d. cross-sectional white-noise panel (the strict null).

    Zero drift, no common factor — nothing forecastable, cross-sectionally or in time.
    The strictest honest-null testbed, driving the anti-overfit regression. Shape
    ``(768, 6)``, seeded.
    """
    return pure_noise_panel(n_obs=_N_OBS, n_assets=_N_ASSETS, seed=_SEED).returns
