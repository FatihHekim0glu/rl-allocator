"""Subprocess guard: the ONNX policy serve path imports NO torch / sb3 / gymnasium.

The hard serve-purity rule for the agents layer: importing and EXERCISING the ONNX
policy serve path (:mod:`rlallocator.agents.onnx_policy`) must never pull torch,
stable-baselines3, or gymnasium into ``sys.modules`` — only numpy + onnxruntime (the
lean ``[serve]`` extra). This runs the check in a fresh SUBPROCESS (immune to what the
parent test session already imported), and exercises a real forward pass through a
tiny ONNX policy (obs -> per-asset scores -> long-only weight simplex) so the guard
covers the LIVE inference path, not just the import.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

#: Heavy training-only modules that must NEVER load on the serve path.
_FORBIDDEN_ON_SERVE: tuple[str, ...] = ("torch", "stable_baselines3", "gymnasium")

#: Skip the real-artifact subprocess tests where the [serve] extras are absent. Use
#: ``find_spec`` (no import) so collecting this module never pulls onnxruntime / onnx
#: into the PARENT test process — only the spawned subprocesses load them, keeping the
#: in-process serve-purity guards in other suites a clean ``sys.modules`` check.
_HAS_SERVE_DEPS: bool = bool(
    importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime")
)
_skip_without_serve = pytest.mark.skipif(
    not _HAS_SERVE_DEPS, reason="onnx / onnxruntime ([serve] extra) not installed"
)


@pytest.mark.unit
def test_import_onnx_policy_loads_no_torch_in_fresh_interpreter() -> None:
    """A bare ``import rlallocator.agents.onnx_policy`` loads none of torch/sb3/gymnasium."""
    forbidden = ", ".join(repr(m) for m in _FORBIDDEN_ON_SERVE)
    code = (
        "import sys\n"
        "import rlallocator.agents.onnx_policy  # noqa: F401\n"
        f"forbidden = [{forbidden}]\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "assert not leaked, f'onnx_policy import leaked: {leaked}'\n"
        "print('ONNX_POLICY_IMPORT_PURE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"onnx_policy import-purity check failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "ONNX_POLICY_IMPORT_PURE_OK" in result.stdout


@pytest.mark.unit
@_skip_without_serve
def test_onnx_policy_forward_pass_loads_no_torch_in_fresh_interpreter(tmp_path: Path) -> None:
    """A full ONNX serve forward pass (load + predict -> simplex weights) loads no torch.

    Builds a tiny dense (10 -> 4) policy graph with onnx (NO torch), then drives a real
    obs -> per-asset scores -> long-only simplex weight forward pass through
    onnxruntime, asserting (a) every served weight row is a valid simplex and (b) no
    torch / sb3 / gymnasium module appears in ``sys.modules`` afterwards.
    """
    artifact = tmp_path / "policy.onnx"
    forbidden = ", ".join(repr(m) for m in _FORBIDDEN_ON_SERVE)
    code = (
        "import sys\n"
        "import numpy as np\n"
        "import onnx\n"
        "from onnx import TensorProto, helper, numpy_helper\n"
        "obs_dim, n_assets = 10, 4\n"
        "rng = np.random.default_rng(0)\n"
        "W = rng.standard_normal((obs_dim, n_assets)).astype('float32')\n"
        "b = rng.standard_normal(n_assets).astype('float32')\n"
        "obs = helper.make_tensor_value_info('observation', TensorProto.FLOAT, ['batch', obs_dim])\n"
        "out = helper.make_tensor_value_info('asset_scores', TensorProto.FLOAT, "
        "['batch', n_assets])\n"
        "g = helper.make_graph(\n"
        "    [helper.make_node('MatMul', ['observation', 'W'], ['mm']),\n"
        "     helper.make_node('Add', ['mm', 'b'], ['asset_scores'])],\n"
        "    'policy', [obs], [out],\n"
        "    [numpy_helper.from_array(W, name='W'), numpy_helper.from_array(b, name='b')])\n"
        "m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])\n"
        "m.ir_version = 10\n"
        f"onnx.save(m, {str(artifact)!r})\n"
        "from rlallocator.agents.onnx_policy import OnnxPolicy, score_weights_from_onnx\n"
        f"weights = score_weights_from_onnx(rng.standard_normal((5, obs_dim)), "
        f"artifact_path={str(artifact)!r})\n"
        "assert weights.shape == (5, n_assets)\n"
        "assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-9), weights.sum(axis=1)\n"
        "assert (weights >= -1e-12).all(), weights.min()\n"
        f"scores = OnnxPolicy({str(artifact)!r}).predict_scores(np.zeros((5, obs_dim)))\n"
        "assert scores.shape == (5, n_assets)\n"
        f"forbidden = [{forbidden}]\n"
        "leaked = sorted(mm for mm in forbidden if mm in sys.modules)\n"
        "assert not leaked, f'serve forward pass leaked: {leaked}'\n"
        "print('ONNX_SERVE_FORWARD_PURE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"serve forward-pass purity check failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "ONNX_SERVE_FORWARD_PURE_OK" in result.stdout


@pytest.mark.unit
@_skip_without_serve
def test_real_onnxruntime_error_paths_out_of_process(tmp_path: Path) -> None:
    """The REAL onnxruntime corrupt-graph + wrong-shape failures normalize to ArtifactError.

    Runs the genuine onnxruntime load/run error paths in a SUBPROCESS so the real
    engine is exercised (corrupt artifact -> load ArtifactError; obs-dim mismatch ->
    run ArtifactError) without importing onnxruntime into the parent test process.
    """
    corrupt = tmp_path / "corrupt.onnx"
    artifact = tmp_path / "policy.onnx"
    code = (
        "import numpy as np\n"
        "import onnx\n"
        "from onnx import TensorProto, helper, numpy_helper\n"
        "from rlallocator._exceptions import ArtifactError\n"
        "from rlallocator.agents.onnx_policy import OnnxPolicy\n"
        f"corrupt = {str(corrupt)!r}\n"
        "open(corrupt, 'wb').write(b'not a valid onnx graph')\n"
        "try:\n"
        "    OnnxPolicy(corrupt).load()\n"
        "    raise SystemExit('corrupt artifact did not raise')\n"
        "except ArtifactError as exc:\n"
        "    assert 'failed to initialize' in str(exc), exc\n"
        "obs_dim, n_assets = 10, 4\n"
        "rng = np.random.default_rng(0)\n"
        "W = rng.standard_normal((obs_dim, n_assets)).astype('float32')\n"
        "b = rng.standard_normal(n_assets).astype('float32')\n"
        "obs = helper.make_tensor_value_info('observation', TensorProto.FLOAT, ['batch', obs_dim])\n"
        "out = helper.make_tensor_value_info('asset_scores', TensorProto.FLOAT, "
        "['batch', n_assets])\n"
        "g = helper.make_graph(\n"
        "    [helper.make_node('MatMul', ['observation', 'W'], ['mm']),\n"
        "     helper.make_node('Add', ['mm', 'b'], ['asset_scores'])],\n"
        "    'policy', [obs], [out],\n"
        "    [numpy_helper.from_array(W, name='W'), numpy_helper.from_array(b, name='b')])\n"
        "m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])\n"
        "m.ir_version = 10\n"
        f"onnx.save(m, {str(artifact)!r})\n"
        "try:\n"
        f"    OnnxPolicy({str(artifact)!r}).predict_scores(np.zeros((2, 7)))\n"
        "    raise SystemExit('wrong-obs-dim did not raise')\n"
        "except ArtifactError as exc:\n"
        "    assert 'onnxruntime run failed' in str(exc), exc\n"
        "print('ONNX_ERROR_PATHS_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"real onnxruntime error-path check failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "ONNX_ERROR_PATHS_OK" in result.stdout
