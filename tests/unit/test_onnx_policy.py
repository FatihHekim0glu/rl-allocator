"""Torch-free, onnxruntime-free unit tests for the ONNX policy serve wrapper.

These exercise :class:`rlallocator.agents.onnx_policy.OnnxPolicy`'s logic + validation
WITHOUT importing the real onnxruntime engine — by injecting a minimal
onnxruntime-session stand-in (the proven rl-trader / gnn-stocks pattern). Keeping the
unit suite free of an in-process ``onnxruntime`` import means it stays a pure
``sys.modules`` check for the other in-process purity guards. The REAL end-to-end
onnxruntime forward pass (a committed-style ONNX artifact loaded + run, scores ->
simplex weights) is covered in the subprocess test
:mod:`tests.unit.test_serve_no_torch` and the slow ONNX-vs-torch parity suite, so the
artifact path is genuinely exercised — just out-of-process.

Covered here: the obs -> per-asset scores forward-pass SHAPE; the scores -> valid
SIMPLEX weight path projection (sum to one, long-only every bar); the lazy idempotent
session; the convenience :func:`score_weights_from_onnx` wrapper; the default-artifact
path; and the :class:`ArtifactError` validation paths (missing artifact before any
engine import, non-finite / mis-shaped / empty input, a run failure normalized to
``ArtifactError``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rlallocator._exceptions import ArtifactError
from rlallocator._validation import is_simplex
from rlallocator.agents.onnx_policy import (
    OnnxPolicy,
    default_artifact_path,
    score_weights_from_onnx,
)

_OBS_DIM = 10
_N_ASSETS = 4


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class _LinearFakeSession:
    """A minimal onnxruntime-session stand-in computing the SAME ``obs @ W + b`` head.

    Stands in for a committed linear policy artifact so the serve wrapper's forward
    pass, shape contract, and score-projection are exercised faithfully WITHOUT
    loading the real onnxruntime engine into the test process (keeping the unit suite
    a clean ``sys.modules`` purity check). It honours the float32 export boundary.
    """

    def __init__(
        self, weight: np.ndarray, bias: np.ndarray, input_name: str = "observation"
    ) -> None:
        self._weight = weight.astype("float32")
        self._bias = bias.astype("float32")
        self._inputs = [_Input(input_name)]

    def get_inputs(self) -> list[_Input]:
        return self._inputs

    def run(self, _outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = feeds[self._inputs[0].name]
        return [(x @ self._weight + self._bias).astype("float32")]


def _wrap(weight: np.ndarray, bias: np.ndarray) -> OnnxPolicy:
    """An ``OnnxPolicy`` with an injected linear fake session (no real onnxruntime)."""
    policy = OnnxPolicy(artifact_path="/unused.onnx")
    policy._session = _LinearFakeSession(weight, bias)
    return policy


@pytest.fixture
def score_weights() -> tuple[np.ndarray, np.ndarray]:
    """A seeded ``(10 -> 4)`` per-asset score linear head."""
    rng = np.random.default_rng(20260621)
    weight = rng.standard_normal((_OBS_DIM, _N_ASSETS)).astype("float64")
    bias = rng.standard_normal(_N_ASSETS).astype("float64")
    return weight, bias


@pytest.mark.unit
def test_default_artifact_path_points_at_shipped_policy() -> None:
    """The default artifact path resolves to ``<pkg>/artifacts/policy.onnx``."""
    path = default_artifact_path()
    assert path.name == "policy.onnx"
    assert path.parent.name == "artifacts"
    assert isinstance(path, Path)


@pytest.mark.unit
def test_predict_scores_matches_numpy_reference(
    score_weights: tuple[np.ndarray, np.ndarray],
) -> None:
    """The forward pass reproduces the ``obs @ W + b`` per-asset score head (float64)."""
    weight, bias = score_weights
    policy = _wrap(weight, bias)
    obs = np.linspace(-1.0, 1.0, _OBS_DIM * 4).reshape(4, _OBS_DIM)
    scores = policy.predict_scores(obs)
    assert scores.shape == (4, _N_ASSETS)
    assert scores.dtype == np.float64
    np.testing.assert_allclose(scores, obs @ weight + bias, atol=1e-5)


@pytest.mark.unit
def test_predict_weights_rows_are_valid_simplices(
    score_weights: tuple[np.ndarray, np.ndarray],
) -> None:
    """Every row of the served weight path is a valid long-only simplex (sum to one, >= 0)."""
    weight, bias = score_weights
    policy = _wrap(weight, bias)
    obs = np.linspace(-3.0, 3.0, _OBS_DIM * 7).reshape(7, _OBS_DIM)
    weights = policy.predict_weights(obs)
    assert weights.shape == (7, _N_ASSETS)
    # Sum to one and non-negative on EVERY bar (the simplex invariant by construction).
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    assert (weights >= -1e-12).all()
    assert all(is_simplex(row, long_only=True) for row in weights)


@pytest.mark.unit
def test_predict_weights_single_observation_vector(
    score_weights: tuple[np.ndarray, np.ndarray],
) -> None:
    """A 1-D observation is treated as a 1-row batch and yields a single simplex row."""
    weight, bias = score_weights
    policy = _wrap(weight, bias)
    weights = policy.predict_weights(np.zeros(_OBS_DIM))
    assert weights.shape == (1, _N_ASSETS)
    assert is_simplex(weights[0], long_only=True)


@pytest.mark.unit
def test_predict_scores_single_observation_vector(
    score_weights: tuple[np.ndarray, np.ndarray],
) -> None:
    """A 1-D observation passed to predict_scores is reshaped to a 1-row batch."""
    weight, bias = score_weights
    policy = _wrap(weight, bias)
    scores = policy.predict_scores(np.zeros(_OBS_DIM))
    assert scores.shape == (1, _N_ASSETS)
    np.testing.assert_allclose(scores[0], bias, atol=1e-5)


@pytest.mark.unit
def test_load_is_lazy_and_idempotent_with_injected_session(
    score_weights: tuple[np.ndarray, np.ndarray],
) -> None:
    """An already-initialized session short-circuits ``load`` (idempotent, no re-init)."""
    weight, bias = score_weights
    policy = _wrap(weight, bias)
    session = policy._session
    assert session is not None
    # load() is a no-op when a session already exists (idempotency contract).
    policy.load()
    assert policy._session is session


@pytest.mark.unit
def test_score_weights_from_onnx_wrapper_matches_class(
    score_weights: tuple[np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The convenience wrapper returns the same weight path as the class path."""
    weight, bias = score_weights
    obs = np.linspace(-2.0, 2.0, _OBS_DIM * 3).reshape(3, _OBS_DIM)
    # Patch OnnxPolicy.load to inject the fake session so the wrapper never touches the
    # filesystem or real onnxruntime.
    fake = _LinearFakeSession(weight, bias)

    def _inject(self: OnnxPolicy) -> OnnxPolicy:
        self._session = fake
        return self

    monkeypatch.setattr(OnnxPolicy, "load", _inject)
    via_wrapper = score_weights_from_onnx(obs, artifact_path="/unused.onnx")
    via_class = _wrap(weight, bias).predict_weights(obs)
    np.testing.assert_allclose(via_wrapper, via_class, atol=1e-12)


@pytest.mark.unit
def test_load_missing_artifact_raises_artifact_error(tmp_path: Path) -> None:
    """A missing artifact raises ``ArtifactError`` (the file check fires before any run).

    ``load`` checks file existence BEFORE importing onnxruntime, so a missing artifact
    never loads an inference engine; the ``not found`` message pins that early-exit
    contract. (The out-of-process subprocess purity test asserts the no-engine-loaded
    property at the process level.)
    """
    missing = tmp_path / "nope.onnx"
    policy = OnnxPolicy(missing)
    with pytest.raises(ArtifactError, match="not found"):
        policy.load()


@pytest.mark.unit
def test_artifact_path_property_round_trips(tmp_path: Path) -> None:
    """The resolved artifact path is exposed verbatim via the property."""
    explicit = tmp_path / "custom.onnx"
    assert OnnxPolicy(explicit).artifact_path == explicit
    assert OnnxPolicy().artifact_path == default_artifact_path()


@pytest.mark.unit
def test_predict_rejects_non_finite_observations(
    score_weights: tuple[np.ndarray, np.ndarray],
) -> None:
    """A NaN/inf observation is rejected as an ``ArtifactError`` at the boundary."""
    weight, bias = score_weights
    policy = _wrap(weight, bias)
    bad = np.zeros((1, _OBS_DIM))
    bad[0, 0] = np.nan
    with pytest.raises(ArtifactError, match="finite"):
        policy.predict_scores(bad)


@pytest.mark.unit
def test_predict_rejects_empty_observations(
    score_weights: tuple[np.ndarray, np.ndarray],
) -> None:
    """An empty observation batch is rejected as an ``ArtifactError``."""
    weight, bias = score_weights
    policy = _wrap(weight, bias)
    with pytest.raises(ArtifactError, match="at least one row"):
        policy.predict_scores(np.zeros((0, _OBS_DIM)))


@pytest.mark.unit
def test_predict_rejects_3d_observations(
    score_weights: tuple[np.ndarray, np.ndarray],
) -> None:
    """A 3-D observation tensor is rejected as an ``ArtifactError``."""
    weight, bias = score_weights
    policy = _wrap(weight, bias)
    with pytest.raises(ArtifactError, match="1-D or 2-D"):
        policy.predict_scores(np.zeros((2, 2, _OBS_DIM)))


class _RaisingSession:
    """A session stand-in whose ``run`` raises (the run-failure normalization path)."""

    def get_inputs(self) -> list[_Input]:
        return [_Input("observation")]

    def run(self, _outputs: Any, _feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        raise RuntimeError("boom from the fake onnxruntime session")


@pytest.mark.unit
def test_predict_normalizes_run_failure_to_artifact_error() -> None:
    """An onnxruntime ``run`` failure is normalized to ``ArtifactError`` (injected session)."""
    policy = OnnxPolicy("/unused.onnx")
    policy._session = _RaisingSession()
    with pytest.raises(ArtifactError, match="onnxruntime run failed"):
        policy.predict_scores(np.zeros((1, _OBS_DIM)))


class _OneDOutputSession:
    """A session returning a flat ``(batch * n_assets,)`` vector (the reshape path)."""

    def __init__(self, n_assets: int) -> None:
        self._n_assets = n_assets
        self._inputs = [_Input("observation")]

    def get_inputs(self) -> list[_Input]:
        return self._inputs

    def run(self, _outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = feeds[self._inputs[0].name]
        rows = x.shape[0]
        flat = np.tile(np.arange(self._n_assets, dtype="float32"), rows)
        return [flat]


@pytest.mark.unit
def test_predict_scores_reshapes_1d_output() -> None:
    """A 1-D ONNX output is reshaped to ``(batch, n_assets)`` (defensive output handling)."""
    policy = OnnxPolicy("/unused.onnx")
    policy._session = _OneDOutputSession(_N_ASSETS)
    scores = policy.predict_scores(np.zeros((3, _OBS_DIM)))
    assert scores.shape == (3, _N_ASSETS)
