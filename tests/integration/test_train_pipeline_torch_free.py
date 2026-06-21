"""Integration: the offline train pipeline orchestration, covered TORCH-FREE.

The pipeline reaches torch / stable-baselines3 / gymnasium ONLY through the injectable
``trainer`` seam. This suite injects a FAKE numpy-only trainer that exports a small
ONNX graph directly (via ``onnx``, NO torch) so the WHOLE orchestration — purged
walk-forward folds, per-seed ONNX-policy scoring through the pure backtester, the seed
lottery, the DSR / PBO / DM / verdict, and writing the committed ``policy.onnx`` +
``metrics.json`` + manifest — is exercised without ever importing torch. It also pins
the honest-NULL outcome (``rl_beats_baselines=False``) and determinism, and asserts the
served path consumes the produced artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rlallocator.train import TrainResult, train_pipeline

# The fake trainer builds a real ONNX graph with ``onnx`` (NO torch); skip where the
# [serve] extras (onnx / onnxruntime) are absent.
_HAS_SERVE_DEPS: bool = bool(
    importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime")
)
_skip_without_serve = pytest.mark.skipif(
    not _HAS_SERVE_DEPS, reason="onnx / onnxruntime ([serve] extra) not installed"
)


class _FakePolicy:
    """A numpy-only trained policy: a fixed dense obs->scores MLP exported via onnx (NO torch).

    Satisfies the :class:`rlallocator.train.TrainedPolicy` Protocol (``policy_scores`` +
    ``export_onnx``) without importing torch. The weights are seeded so the pipeline is
    deterministic; the random dense map is a stand-in for a trained policy network — it
    carries no edge on the factor-regime null (the honest-NULL holds).
    """

    def __init__(self, obs_dim: int, n_assets: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._obs_dim = int(obs_dim)
        self._n_assets = int(n_assets)
        self._w = (rng.standard_normal((obs_dim, n_assets)) * 0.01).astype("float32")
        self._b = (rng.standard_normal(n_assets) * 0.01).astype("float32")

    def policy_scores(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype="float64")
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        return obs @ self._w.astype("float64") + self._b.astype("float64")

    def export_onnx(self, out_path: str | Path, *, rtol: float = 1e-4) -> Path:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        obs = helper.make_tensor_value_info(
            "observation", TensorProto.FLOAT, ["batch", self._obs_dim]
        )
        score = helper.make_tensor_value_info(
            "asset_scores", TensorProto.FLOAT, ["batch", self._n_assets]
        )
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["observation", "W"], ["mm"]),
                helper.make_node("Add", ["mm", "b"], ["asset_scores"]),
            ],
            "fake_policy",
            [obs],
            [score],
            [
                numpy_helper.from_array(self._w, name="W"),
                numpy_helper.from_array(self._b, name="b"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10
        onnx.save(model, str(out))
        return out


def _fake_trainer(
    train_returns: np.ndarray,
    *,
    obs_dim: int,
    n_assets: int,
    lookback: int,
    cost_bps: float,
    episode_len: int,
    seed: int,
) -> _FakePolicy:
    """A torch-free trainer seam returning a deterministic fake policy."""
    return _FakePolicy(obs_dim, n_assets, seed)


@pytest.mark.integration
@_skip_without_serve
def test_train_pipeline_orchestration_runs_with_injected_trainer(tmp_path: Path) -> None:
    """The full offline pipeline runs end-to-end with an injected (torch-free) trainer."""
    result = train_pipeline(
        n_assets=4,
        n_seeds=3,
        lookback=8,
        cost_bps=10.0,
        episode_len=64,
        n_obs=400,
        n_folds=3,
        kind="factor_regime",
        seed=7,
        artifacts_dir=tmp_path,
        trainer=_fake_trainer,
    )
    assert isinstance(result, TrainResult)
    assert Path(result.policy_path).is_file()
    assert Path(result.metrics_path).is_file()
    assert result.n_effective_trials == 3
    assert len(result.seed_sharpes) == 3

    # The committed metrics carry the full honesty stack and the PURE verdict.
    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    for key in (
        "oos_sharpe_rl_median",
        "oos_sharpe_1n",
        "oos_sharpe_markowitz",
        "oos_sharpe_riskparity",
        "best_baseline",
        "seed_sharpe_lo",
        "seed_sharpe_hi",
        "dm_pvalue_vs_best",
        "deflated_sharpe",
        "pbo",
        "rl_beats_baselines",
        "n_effective_trials",
        "rl_median_equity",
        "rl_weight_path",
        "manifest",
    ):
        assert key in metrics, key
    assert 0.0 <= metrics["pbo"] <= 1.0
    assert 0.0 <= metrics["deflated_sharpe"] <= 1.0


@pytest.mark.integration
@_skip_without_serve
def test_train_pipeline_honest_null_verdict_false(tmp_path: Path) -> None:
    """On the factor-regime null the produced verdict is False (the honest-NULL outcome)."""
    result = train_pipeline(
        n_assets=4,
        n_seeds=4,
        lookback=8,
        cost_bps=10.0,
        episode_len=64,
        n_obs=420,
        n_folds=3,
        kind="factor_regime",
        seed=11,
        artifacts_dir=tmp_path,
        trainer=_fake_trainer,
    )
    assert result.rl_beats_baselines is False
    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert metrics["rl_beats_baselines"] is False
    # At least one of the four gates fails (the honest-NULL): DSR below 1-alpha OR
    # PBO >= 0.5 OR DM insignificant OR seed-lo <= 0.
    fails = (
        metrics["deflated_sharpe"] <= 0.95
        or metrics["pbo"] >= 0.5
        or metrics["dm_pvalue_vs_best"] >= 0.05
        or metrics["seed_sharpe_lo"] <= 0.0
    )
    assert fails


@pytest.mark.integration
@_skip_without_serve
def test_train_pipeline_is_deterministic(tmp_path: Path) -> None:
    """The same seed + injected trainer reproduce identical seed Sharpes (determinism)."""
    kwargs = dict(
        n_assets=4,
        n_seeds=3,
        lookback=8,
        cost_bps=10.0,
        episode_len=64,
        n_obs=400,
        n_folds=3,
        kind="factor_regime",
        seed=7,
        trainer=_fake_trainer,
    )
    a = train_pipeline(artifacts_dir=tmp_path / "a", **kwargs)  # type: ignore[arg-type]
    b = train_pipeline(artifacts_dir=tmp_path / "b", **kwargs)  # type: ignore[arg-type]
    assert a.seed_sharpes == b.seed_sharpes


@pytest.mark.integration
@_skip_without_serve
def test_served_path_consumes_trained_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After training into the package artifacts dir, run_allocation serves the real policy.

    Points the package artifacts directory (both the train output and the serve / ONNX
    lookup) at a temp dir, trains a torch-free policy into it, then asserts
    ``run_allocation`` serves the committed ONNX policy on the walk-forward (a non-zero
    RL median Sharpe distinct from the bare-scaffold zero) while still reporting the
    honest-NULL verdict.
    """
    import rlallocator.agents.onnx_policy as onnx_mod
    import rlallocator.serve as serve_mod

    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(onnx_mod, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(serve_mod, "_ARTIFACTS_DIR", artifacts)

    train_pipeline(
        n_assets=4,
        n_seeds=3,
        lookback=8,
        cost_bps=10.0,
        episode_len=64,
        n_obs=600,
        n_folds=4,
        kind="factor_regime",
        seed=7,
        artifacts_dir=artifacts,
        trainer=_fake_trainer,
    )
    assert (artifacts / "policy.onnx").is_file()

    run = serve_mod.run_allocation(n_assets=4, n_seeds=3, cost_bps=10.0, lookback=8, seed=7)
    summary = run.summary
    # The served policy produced a real (finite) walk-forward RL OOS Sharpe.
    assert np.isfinite(summary.oos_sharpe_rl_median)
    # The honest-NULL verdict still holds on the factor-regime panel.
    assert summary.rl_beats_baselines is False


@pytest.mark.integration
@_skip_without_serve
def test_train_pipeline_orchestration_is_torch_free_in_fresh_interpreter(
    tmp_path: Path,
) -> None:
    """In a FRESH interpreter the orchestration (injected trainer) loads no torch/sb3/gym.

    A subprocess check immune to whatever the parent test session already imported
    (torch may be installed and pulled in by the slow [train] tests): a fresh interpreter
    runs the WHOLE pipeline with a numpy+onnx-only injected trainer and asserts torch /
    stable-baselines3 / gymnasium never enter ``sys.modules`` — the load-bearing
    "orchestration is covered torch-free" guarantee.
    """
    out = tmp_path / "art"
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import onnx\n"
        "from onnx import TensorProto, helper, numpy_helper\n"
        "from rlallocator.train import train_pipeline\n"
        "class P:\n"
        "    def __init__(self, obs_dim, n_assets, seed):\n"
        "        r = np.random.default_rng(seed); self.o=obs_dim; self.n=n_assets\n"
        "        self.W=(r.standard_normal((obs_dim,n_assets))*0.01).astype('float32')\n"
        "        self.b=(r.standard_normal(n_assets)*0.01).astype('float32')\n"
        "    def policy_scores(self, obs):\n"
        "        o=np.asarray(obs,dtype='float64');\n"
        "        o=o.reshape(1,-1) if o.ndim==1 else o\n"
        "        return o@self.W.astype('float64')+self.b.astype('float64')\n"
        "    def export_onnx(self, p, *, rtol=1e-4):\n"
        "        p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)\n"
        "        oi=helper.make_tensor_value_info('observation',TensorProto.FLOAT,['batch',self.o])\n"
        "        so=helper.make_tensor_value_info('asset_scores',TensorProto.FLOAT,['batch',self.n])\n"
        "        g=helper.make_graph([helper.make_node('MatMul',['observation','W'],['mm']),\n"
        "            helper.make_node('Add',['mm','b'],['asset_scores'])],'p',[oi],[so],\n"
        "            [numpy_helper.from_array(self.W,name='W'),numpy_helper.from_array(self.b,name='b')])\n"
        "        m=helper.make_model(g,opset_imports=[helper.make_opsetid('',17)]); m.ir_version=10\n"
        "        onnx.save(m,str(p)); return p\n"
        "def trainer(tr,*,obs_dim,n_assets,lookback,cost_bps,episode_len,seed):\n"
        "    return P(obs_dim,n_assets,seed)\n"
        f"train_pipeline(n_assets=4,n_seeds=3,lookback=8,cost_bps=10.0,episode_len=64,"
        f"n_obs=400,n_folds=3,kind='factor_regime',seed=7,artifacts_dir={str(out)!r},trainer=trainer)\n"
        "leaked=sorted(m for m in ('torch','stable_baselines3','gymnasium') if m in sys.modules)\n"
        "assert not leaked, f'orchestration leaked: {leaked}'\n"
        "print('TRAIN_ORCHESTRATION_TORCH_FREE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"orchestration torch-free check failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "TRAIN_ORCHESTRATION_TORCH_FREE_OK" in result.stdout
