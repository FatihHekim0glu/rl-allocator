# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Typed stubs (signatures + docstrings + `NotImplementedError`) for the offline
  `[train]` path: `agents/ppo.py`, `agents/onnx_policy.py`, `train.py`.
- Lazy-plotly figure builders (`plots.py`: equity curves, the weight-allocation area
  chart, the seed-lottery dispersion) and the Typer CLI (`cli.py`).
- Partitioned tests (`unit` / `parity` / `property` / `regression` / `integration`)
  with seeded conftest fixtures (`synthetic_panel`, `learnable_edge`, `pure_noise`) and
  an import-purity smoke test.
- Tooling: `pyproject.toml` extras (`[data]` / `[serve]`=onnxruntime /
  `[train]`=torch+sb3+gymnasium / `[viz]` / `[dev]`; NO `[all]`, NO torch in `[serve]`),
  ruff + strict mypy config, CI (py3.11–3.13, mypy continue-on-error, cov ≥ 85), and a
  no-AI-attribution guard.
