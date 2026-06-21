"""Import-purity smoke test: ``import rlallocator`` pulls in NO heavy dependency.

The hard rule: ``src/rlallocator/`` has ZERO import-time side effects and imports NO
heavy dependency at module load. A fresh interpreter that does ``import rlallocator``
must NOT have ``torch``, ``stable_baselines3``, ``gymnasium``, or ``onnxruntime``
loaded in ``sys.modules`` — those load LAZILY inside their functions, and the serve
path imports ``onnxruntime`` only when an ONNX session is actually created. This test
runs the check in a SUBPROCESS so it is immune to whatever the parent test session has
already imported.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Heavy modules that must NEVER be present after a bare ``import rlallocator``.
_FORBIDDEN_AT_IMPORT: tuple[str, ...] = (
    "torch",
    "stable_baselines3",
    "gymnasium",
    "onnxruntime",
)


@pytest.mark.unit
def test_import_rlallocator_is_import_pure_in_fresh_interpreter() -> None:
    """A fresh ``import rlallocator`` loads none of torch/sb3/gymnasium/onnxruntime."""
    forbidden = ", ".join(repr(m) for m in _FORBIDDEN_AT_IMPORT)
    code = (
        "import sys\n"
        "import rlallocator\n"
        f"forbidden = [{forbidden}]\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "assert not leaked, f'rlallocator import leaked heavy modules: {leaked}'\n"
        "print('IMPORT_PURE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"fresh-interpreter import-purity check failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "IMPORT_PURE_OK" in result.stdout


@pytest.mark.unit
def test_public_api_is_importable() -> None:
    """The curated public API (``rlallocator.__all__``) imports without heavy deps."""
    import rlallocator

    assert rlallocator.__version__ == "0.1.0"
    for name in (
        "PortfolioEnv",
        "vectorized_backtest",
        "derive_verdict",
        "run_allocation",
        "probability_of_backtest_overfitting",
        "markowitz_weights",
    ):
        assert name in rlallocator.__all__
        assert hasattr(rlallocator, name)


@pytest.mark.unit
def test_serve_path_imports_no_torch_in_fresh_interpreter() -> None:
    """Importing the serve entrypoint module loads no torch/sb3/gymnasium."""
    code = (
        "import sys\n"
        "import rlallocator.serve\n"
        "import rlallocator.agents.onnx_policy\n"
        "import rlallocator.agents.baselines\n"
        "leaked = sorted(m for m in ('torch', 'stable_baselines3', 'gymnasium') "
        "if m in sys.modules)\n"
        "assert not leaked, f'serve path leaked: {leaked}'\n"
        "print('SERVE_PURE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"serve-path import-purity check failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "SERVE_PURE_OK" in result.stdout
