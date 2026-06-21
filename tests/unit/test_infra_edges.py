"""Unit tests covering infra edge paths: costs, walk-forward, manifest, rng, dsr, metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rlallocator._exceptions import InsufficientDataError, ValidationError
from rlallocator._manifest import RunManifest, config_hash
from rlallocator._rng import make_rng, spawn_substreams
from rlallocator.costs import TurnoverCost
from rlallocator.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from rlallocator.evaluation.metrics import strategy_metrics
from rlallocator.walk_forward import Fold, make_folds, required_purge


@pytest.mark.unit
def test_turnover_cost_l1() -> None:
    """TurnoverCost charges bps on the L1 weight change."""
    c = TurnoverCost(bps=10.0)
    cost = c.cost(np.array([1.0, 0.0]), np.array([0.0, 1.0]))  # L1 change = 2.0
    assert math.isclose(cost, 2.0 * 10.0 / 10_000.0)


@pytest.mark.unit
def test_turnover_cost_rejects_negative_bps() -> None:
    """A negative bps is rejected at construction."""
    with pytest.raises(ValidationError):
        TurnoverCost(bps=-1.0)


@pytest.mark.unit
def test_turnover_cost_rejects_length_mismatch() -> None:
    """Mismatched weight-vector lengths are rejected."""
    with pytest.raises(ValidationError):
        TurnoverCost(bps=1.0).cost(np.array([1.0]), np.array([0.5, 0.5]))


@pytest.mark.unit
def test_required_purge_and_validation() -> None:
    """required_purge = lookback + horizon - 1; invalid args raise."""
    assert required_purge(64, 1) == 64
    with pytest.raises(ValidationError):
        required_purge(0)


@pytest.mark.unit
def test_make_folds_geometry() -> None:
    """make_folds builds purged/embargoed folds with train_end <= test_start."""
    folds = make_folds(500, lookback=16, n_folds=4)
    assert len(folds) == 4
    for f in folds:
        assert isinstance(f, Fold)
        assert f.train_end <= f.test_start
        assert f.test_start < f.test_end
        assert f.to_dict()["train_start"] == f.train_start


@pytest.mark.unit
def test_make_folds_too_short_raises() -> None:
    """A panel too short to host the folds raises InsufficientDataError."""
    with pytest.raises(InsufficientDataError):
        make_folds(20, lookback=16, n_folds=4)


@pytest.mark.unit
def test_make_folds_rejects_bad_args() -> None:
    """make_folds rejects n_folds < 1 and negative embargo."""
    with pytest.raises(ValidationError):
        make_folds(500, lookback=8, n_folds=0)
    with pytest.raises(ValidationError):
        make_folds(500, lookback=8, embargo=-1)


@pytest.mark.unit
def test_manifest_capture_and_hash() -> None:
    """RunManifest.capture records a config hash + seed; config_hash is order-invariant."""
    m = RunManifest.capture({"a": 1, "b": 2}, seed=7)
    assert m.seed == 7
    assert isinstance(m.git_sha, str)
    assert m.to_dict()["config_hash"] == m.config_hash
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


@pytest.mark.unit
def test_rng_determinism_and_substreams() -> None:
    """make_rng is seed-deterministic; spawn_substreams yields independent generators."""
    assert make_rng(7).integers(0, 1000) == make_rng(7).integers(0, 1000)
    subs = spawn_substreams(7, 3)
    assert len(subs) == 3
    with pytest.raises(ValueError, match="non-negative"):
        make_rng(-1)


@pytest.mark.unit
def test_dsr_single_trial_reduces_to_psr() -> None:
    """With n_trials=1 the DSR equals the plain PSR against zero."""
    psr = probabilistic_sharpe_ratio(0.1, n_obs=300)
    dsr = deflated_sharpe_ratio(0.1, n_obs=300, n_trials=1, variance_of_trial_sharpes=0.02)
    assert math.isclose(psr, dsr, abs_tol=1e-12)


@pytest.mark.unit
def test_dsr_rejects_bad_args() -> None:
    """The DSR rejects n_obs < 2, n_trials < 1, and negative variance."""
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=1, n_trials=5, variance_of_trial_sharpes=0.0)
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=10, n_trials=0, variance_of_trial_sharpes=0.0)
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=10, n_trials=5, variance_of_trial_sharpes=-1.0)


@pytest.mark.unit
def test_strategy_metrics_bundle() -> None:
    """strategy_metrics assembles a consistent bundle from net returns + a weight path."""
    rng = np.random.default_rng(0)
    net = rng.normal(0.0, 0.01, size=50)
    weights = np.full((50, 3), 1.0 / 3.0)
    m = strategy_metrics(net, weights)
    assert m.n_bars == 50
    assert m.max_drawdown <= 0.0
    assert m.turnover >= 0.0
    assert m.to_dict()["n_bars"] == 50


@pytest.mark.unit
def test_strategy_metrics_length_mismatch() -> None:
    """strategy_metrics rejects a net/weights row-count mismatch."""
    with pytest.raises(ValidationError):
        strategy_metrics(np.zeros(10), np.zeros((8, 3)))
