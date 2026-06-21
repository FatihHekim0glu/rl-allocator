"""In-process REAL onnxruntime round-trip tests for the ONNX policy serve wrapper.

Distinct from :mod:`tests.unit.test_onnx_policy` (which injects a fake session to keep
the bulk of the unit suite engine-free) and from :mod:`tests.unit.test_serve_no_torch`
(which runs the genuine engine OUT of process for the no-torch purity guard), this file
loads a real, tiny ONNX policy artifact (built with ``onnx``, NO torch) through the
genuine onnxruntime session IN-process so the live ``OnnxPolicy.load`` /
``predict_scores`` / ``predict_weights`` path — session creation, idempotent reuse,
scores -> simplex projection — is exercised directly. It is SKIPPED where the
``[serve]`` extra (onnx + onnxruntime) is absent.

The in-process import of onnxruntime here is harmless to the project's import-purity
guards: those guards run in fresh SUBPROCESSES, so what this test loads into the parent
process never leaks into them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rlallocator._validation import is_simplex
from rlallocator.agents.onnx_policy import OnnxPolicy, score_weights_from_onnx

_HAS_SERVE: bool = bool(
    importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime")
)
pytestmark = pytest.mark.skipif(
    not _HAS_SERVE, reason="onnx / onnxruntime ([serve] extra) not installed"
)

_OBS_DIM = 10
_N_ASSETS = 4


def _write_linear_policy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write a tiny dense ``obs @ W + b`` ONNX policy (NO torch) and return ``(W, b)``."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(20260621)
    weight = rng.standard_normal((_OBS_DIM, _N_ASSETS)).astype("float32")
    bias = rng.standard_normal(_N_ASSETS).astype("float32")
    obs = helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", _OBS_DIM])
    out = helper.make_tensor_value_info("asset_scores", TensorProto.FLOAT, ["batch", _N_ASSETS])
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["observation", "W"], ["mm"]),
            helper.make_node("Add", ["mm", "b"], ["asset_scores"]),
        ],
        "policy",
        [obs],
        [out],
        [numpy_helper.from_array(weight, name="W"), numpy_helper.from_array(bias, name="b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))
    return weight.astype("float64"), bias.astype("float64")


@pytest.mark.unit
def test_real_onnxruntime_predict_scores_matches_reference(tmp_path: Path) -> None:
    """The real onnxruntime forward pass reproduces ``obs @ W + b`` to float tolerance."""
    artifact = tmp_path / "policy.onnx"
    weight, bias = _write_linear_policy(artifact)
    obs = np.linspace(-1.0, 1.0, _OBS_DIM * 5).reshape(5, _OBS_DIM)
    scores = OnnxPolicy(artifact).predict_scores(obs)
    assert scores.shape == (5, _N_ASSETS)
    np.testing.assert_allclose(scores, obs @ weight + bias, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_real_onnxruntime_predict_weights_are_simplices(tmp_path: Path) -> None:
    """The real served weight path is a valid long-only simplex every bar."""
    artifact = tmp_path / "policy.onnx"
    _write_linear_policy(artifact)
    obs = np.linspace(-3.0, 3.0, _OBS_DIM * 6).reshape(6, _OBS_DIM)
    weights = score_weights_from_onnx(obs, artifact_path=artifact)
    assert weights.shape == (6, _N_ASSETS)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    assert all(is_simplex(row, long_only=True) for row in weights)


@pytest.mark.unit
def test_real_onnxruntime_load_is_idempotent(tmp_path: Path) -> None:
    """A second ``load`` reuses the live onnxruntime session (idempotency on the real path)."""
    artifact = tmp_path / "policy.onnx"
    _write_linear_policy(artifact)
    policy = OnnxPolicy(artifact).load()
    session = policy._session
    assert session is not None
    policy.load()
    assert policy._session is session
