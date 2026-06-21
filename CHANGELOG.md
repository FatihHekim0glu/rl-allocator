# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation ship: `docs/DESIGN.md` (the system design — causal env, parity oracle,
  walk-forward-in-served-path wiring, offline-train → ONNX-serve split, and the honesty
  stack) and `docs/decisions/` Architecture Decision Records (simplex-weight action,
  causal next-bar reward, parity oracle, walk-forward served path, DSR confidence gate,
  ONNX-policy serve).
- Offline training pipeline (`train.py`) wired end-to-end behind an INJECTABLE
  `trainer` seam: synthetic panel → purged/embargoed walk-forward → per-seed policy
  train → ONNX export → each FROZEN ONNX policy scored on the concatenated purged OOS
  folds (pure numpy, no test-time learning) → seed lottery + DM-vs-best + Deflated
  Sharpe + CSCV PBO (`n_trials = #seeds × #HP`) → the PURE `rl_beats_baselines`
  verdict → committed `artifacts/{policy.onnx, metrics.json}` + a `RunManifest`. torch /
  stable-baselines3 / gymnasium are reached ONLY through the seam, so the whole
  orchestration is covered TORCH-FREE by injecting a numpy+onnx trainer (a fresh-
  interpreter subprocess test pins it).
- Committed offline-trained artifacts: a real SB3-PPO policy MLP exported to ONNX
  (`artifacts/policy.onnx`, < 10 MB) + the precomputed honest-NULL `metrics.json`
  (`scripts/train_committed_policy.py` reproduces them).
- The served path now computes the headline RL OOS metrics from the SAME purged
  walk-forward folds as the baselines: `serve.run_allocation` serves the committed ONNX
  policy on each fold's OOS block (onnxruntime, NO torch) and RE-DERIVES the PURE
  verdict from the live walk-forward Diebold-Mariano + the committed-offline DSR /
  seed-lower-bound / PBO gates. Absent a committed policy it degrades to the honest-NULL
  placeholder.

### Changed
- README: replaced the Validation placeholder with the ACTUAL committed metrics
  (`artifacts/metrics.json`) — RL median OOS Sharpe, the three baseline Sharpes, the best
  baseline, the across-seed band, DM p-value, Deflated Sharpe, PBO, and
  `rl_beats_baselines = false`; added a correctness-gates table (parity 1e-10, ONNX↔torch
  1e-4, DSR 1e-10, DM, PBO/CSCV, causal obs/reward, valid simplex, walk-forward-in-served-
  path, learnable_edge sanity, honest null), a Reproduce block (lean install +
  train/backtest/compare + EODHD/Polygon real-data path + ruff/mypy/pytest gates), and
  tightened Limitations.
- `agents/ppo.py` / `agents/onnx_policy.py` / `train.py` graduated from typed
  `NotImplementedError` stubs to the implemented offline path (SB3 PPO → policy-MLP
  extraction → ONNX export with a 1e-4 torch-vs-ONNX parity gate → onnxruntime serve).

### Initial scaffold
- Initial scaffold of `rl-allocator`: a leakage-free, overfit-aware multi-asset RL
  portfolio allocator with an honest-NULL deliverable.
- Import-pure, strictly-typed `src/rlallocator/` package (src-layout, `py.typed`).
- Core infra: `_validation` (with the simplex helpers `is_simplex` /
  `project_to_simplex`), `_constants`, `_typing`, `_exceptions`
  (`RlAllocatorError` base + `ArtifactError` + `ParityError`), `_manifest`, `_rng`.
- Causal multi-asset portfolio env (`env/portfolio_env.py`: weight-simplex action,
  strictly next-bar reward), the vectorized multi-asset backtester
  (`env/backtester.py`), and the parity oracle (`env/parity.py`: vectorized ==
  stepwise to 1e-10, with a deliberately-leaky negative control it catches).
- Pure-numpy allocation baselines (`agents/baselines.py`: equal-weight 1/N, Markowitz
  minimum-variance, risk-parity) computed live with train-only covariance.
- The PURE honesty kernels — fully implemented: Probabilistic / Deflated Sharpe
  (`evaluation/dsr.py`), Diebold-Mariano (`evaluation/diebold_mariano.py`), the CSCV
  Probability of Backtest Overfitting (`evaluation/pbo.py`), the seed lottery
  (`evaluation/seed_lottery.py`), the OOS metrics (`evaluation/metrics.py`), and the
  PURE `rl_beats_baselines` verdict (`evaluation/verdict.py`, gated at DM-favours AND
  DSR > 1−alpha AND seed-lo > 0 AND PBO < 0.5).
- The purged/embargoed walk-forward (`walk_forward.py`), wired into the served path
  (`serve.run_allocation` computes the headline OOS metrics from the concatenated
  purged folds, not the full sample).
- Synthetic multi-asset data: a factor + regime-switching panel (the honest null), a
  `learnable_edge` sanity fixture, and a `pure_noise` strict null (`data/synthetic.py`),
  plus the lazy loaders + the vendored Polygon provider (`data/loaders.py`,
  `data_providers/polygon.py`).
- The offline `[train]` path modules `agents/ppo.py` (SB3 PPO wrapper + policy-MLP
  extraction + ONNX export), `agents/onnx_policy.py` (onnxruntime serve), and `train.py`
  (the pipeline orchestration).
- Lazy-plotly figure builders (`plots.py`: equity curves, the weight-allocation area
  chart, the seed-lottery dispersion) and the Typer CLI (`cli.py`).
- Partitioned tests (`unit` / `parity` / `property` / `regression` / `integration`)
  with seeded conftest fixtures (`synthetic_panel`, `learnable_edge`, `pure_noise`) and
  an import-purity smoke test.
- Tooling: `pyproject.toml` extras (`[data]` / `[serve]`=onnxruntime /
  `[train]`=torch+sb3+gymnasium / `[viz]` / `[dev]`; NO `[all]`, NO torch in `[serve]`),
  ruff + strict mypy config, CI (py3.11–3.13, mypy continue-on-error, cov ≥ 85), and a
  no-AI-attribution guard.
