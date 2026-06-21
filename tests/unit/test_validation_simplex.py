"""Unit tests for the validation guardrails + the simplex helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rlallocator._exceptions import InsufficientDataError, ValidationError
from rlallocator._validation import (
    ensure_dataframe,
    ensure_series,
    is_simplex,
    project_to_simplex,
    validate_min_obs,
)


@pytest.mark.unit
def test_ensure_series_rejects_nan() -> None:
    """A NaN in a series is rejected unless allow_nan."""
    with pytest.raises(ValidationError):
        ensure_series([1.0, np.nan, 2.0])
    assert ensure_series([1.0, np.nan], allow_nan=True).isna().any()


@pytest.mark.unit
def test_ensure_dataframe_shape_guard() -> None:
    """ensure_dataframe coerces a 2-D ndarray and rejects 1-D."""
    frame = ensure_dataframe(np.zeros((4, 3)))
    assert frame.shape == (4, 3)
    with pytest.raises(ValidationError):
        ensure_dataframe(np.zeros(5))


@pytest.mark.unit
def test_validate_min_obs() -> None:
    """validate_min_obs raises when there are too few rows."""
    frame = ensure_dataframe(np.zeros((2, 4)))
    with pytest.raises(InsufficientDataError):
        validate_min_obs(frame, 5)


@pytest.mark.unit
def test_is_simplex_accepts_valid_and_rejects_invalid() -> None:
    """is_simplex accepts a valid long-only simplex and rejects negatives / off-budget."""
    assert is_simplex([0.25, 0.25, 0.5])
    assert not is_simplex([0.5, 0.6])  # sums to 1.1
    assert not is_simplex([-0.1, 1.1])  # negative weight, long-only
    assert is_simplex([-0.1, 1.1], long_only=False)  # budget holds; shorts allowed


@pytest.mark.unit
def test_project_to_simplex_produces_valid_simplex() -> None:
    """project_to_simplex maps any score vector onto a valid long-only simplex."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        scores = rng.normal(size=8)
        w = project_to_simplex(scores)
        assert is_simplex(w)
        assert math.isclose(float(w.sum()), 1.0, abs_tol=1e-9)
        assert bool((w >= 0.0).all())


@pytest.mark.unit
def test_project_to_simplex_already_simplex_is_identity() -> None:
    """A vector that is already a simplex projects (near-)onto itself."""
    w0 = np.array([0.2, 0.3, 0.5])
    w = project_to_simplex(w0)
    assert np.allclose(w, w0, atol=1e-9)


@pytest.mark.unit
def test_project_to_simplex_rejects_nonfinite() -> None:
    """A non-finite score vector is rejected."""
    with pytest.raises(ValidationError):
        project_to_simplex([np.nan, 1.0])
