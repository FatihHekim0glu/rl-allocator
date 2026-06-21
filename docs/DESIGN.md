# Design — rl-allocator

`rl-allocator` trains a PPO agent to allocate across a multi-asset basket (a simplex of
portfolio weights) in a cost-aware environment, then evaluates it out-of-sample inside a
purged walk-forward against equal-weight, Markowitz and risk-parity baselines. The
deliverable is **the honesty stack**, not a profit claim: the documented result is that
the agent does **not** reliably beat the baselines net of costs, and the apparatus proves
that conclusion is leakage-free, parity-checked, and overfit-aware.

This document explains how the pieces fit. Per-decision rationale lives in
[`docs/decisions/`](decisions).

## 1. Goals and non-goals

**Goals**

- A causal, cost-aware, multi-asset portfolio env whose weights are a valid simplex every
  bar and whose reward is strictly next-bar (no look-ahead).
- A vectorized backtester that is provably identical to a step-by-step env rollout (the
  parity oracle), with a deliberately-leaky negative control it must catch.
- An out-of-sample evaluation that runs inside a purged/embargoed walk-forward, with the
  walk-forward **wired into the served path** (not just the training script).
- A multiplicity-aware honesty layer: seed-lottery dispersion, Diebold-Mariano vs. the
  best baseline, the Deflated Sharpe at a confidence level, and CSCV PBO — combined into
  one pure verdict.
- A lean serve path that imports `onnxruntime` only (never torch / sb3 / gymnasium) and
  never trains per request.

**Non-goals**

- Beating the baselines. (The synthetic default is constructed so no allocation beats
  1/N net of costs — the honest null.)
- Live execution. Costs are a turnover penalty; there is no broker, market impact, or
  liquidity model.
- A `[all]` extra or any heavy import at module load. `import rlallocator` is import-pure.

## 2. Package layout

```
src/rlallocator/
  _validation.py _constants.py _typing.py _exceptions.py _manifest.py _rng.py  # infra
  data/synthetic.py          # factor+regime panel, learnable_edge, pure_noise
  data/loaders.py            # synthetic default; Polygon/EODHD optional, lazy imports
  data_providers/polygon.py  # vendored Polygon EOD provider (lazy httpx)
  env/portfolio_env.py       # causal simplex-weight env, next-bar reward
  env/backtester.py          # vectorized multi-asset equity-curve evaluator
  env/parity.py              # vectorized == stepwise (1e-10) + leaky negative control
  costs.py                   # turnover (L1) cost model
  agents/baselines.py        # 1/N, Markowitz, risk-parity (pure numpy, train-only cov)
  agents/ppo.py              # SB3 PPO + policy-MLP extraction + ONNX export  [train]
  agents/onnx_policy.py      # onnxruntime serve -> simplex weights  (NO torch)
  evaluation/metrics.py      # OOS Sharpe, max drawdown, turnover
  evaluation/seed_lottery.py # across-seed median + lower bound
  evaluation/diebold_mariano.py  # DM vs best baseline (HAC)
  evaluation/dsr.py          # probabilistic / deflated Sharpe
  evaluation/pbo.py          # CSCV probability of backtest overfitting
  evaluation/verdict.py      # the PURE rl_beats_baselines verdict
  walk_forward.py            # purged/embargoed folds
  train.py                   # OFFLINE: train -> ONNX -> precompute metrics  [train]
  serve.py                   # run_allocation: live baselines + committed ONNX, walk-forward
  plots.py                   # lazy plotly figures
  cli.py                     # train / backtest / compare
  artifacts/{policy.onnx, metrics.json}  # committed offline deliverables
```

Import purity is enforced by `tests/unit/test_import_purity.py`: torch / sb3 / gymnasium
/ onnx are reached only lazily (inside functions), so a lean `[serve]` install can
`import rlallocator` and serve without them.

## 3. The causal environment and the parity oracle

`env/portfolio_env.py` is the source of truth for the reward dynamics. The action is a
raw per-asset score vector projected onto the long-only simplex via
`_validation.project_to_simplex`, so the held weights satisfy `Σw = 1, w ≥ 0` **every
bar** by construction (ADR-0001). The reward at bar `t` is

```
reward_t = w_t · r_{t -> t+1} - (cost_bps / 1e4) * ‖w_t - w_{t-1}‖_1
```

— the weights chosen at `t` earn the **next** bar's returns, and the observation at `t`
uses only data `≤ t` (ADR-0002). A Hypothesis property test perturbs future bars and
asserts the obs/reward at `t` are invariant.

`env/backtester.py` is a vectorized equity-curve evaluator over a whole weight path.
`env/parity.py` asserts the vectorized backtester equals the step-by-step env rollout to
**1e-10** for arbitrary weight paths, and ships `leaky_backtest`, a deliberately
look-ahead variant the oracle is asserted to **catch** (ADR-0003). This is the
look-ahead tripwire: any change that lets future information leak into the equity curve
breaks parity.

## 4. Baselines (live, train-only covariance)

`agents/baselines.py` implements equal-weight 1/N, Markowitz mean-variance, and
risk-parity in pure numpy. Markowitz and risk-parity estimate their covariance on the
**train fold only** — never the full sample — so there is no look-ahead, pinned by unit
tests against hand references. The baselines run **live** on every request (they are
cheap); only the RL policy is precomputed.

## 5. Walk-forward, wired into the served path

`walk_forward.py` builds anchored, purged, embargoed folds. The purge size is
`required_purge(lookback, horizon) = lookback + horizon - 1`, which removes every
train/test boundary bar whose label or look-back window straddles the split; the embargo
(default 1) holds out the return-horizon bars after the train window (de Prado 2018).

The recurring bug across prior projects is that the *training script* uses the rigorous
walk-forward but the *served entrypoint* silently computes metrics on the full sample.
Here `serve.run_allocation` computes the headline OOS metrics — for the baselines **and**
the committed ONNX policy — from the **concatenated purged OOS folds**, and a regression
test (`tests/regression/test_served_path_walk_forward.py`) asserts the served path
actually calls the walk-forward function (ADR-0004).

## 6. The honesty stack and the pure verdict

The OOS evaluation produces, per seed:

- **OOS net Sharpe, max drawdown, turnover** (`evaluation/metrics.py`).
- **Seed lottery** — the across-seed median Sharpe and a lower bound
  (`evaluation/seed_lottery.py`).
- **Diebold-Mariano** — the median-seed net-return series vs. the best baseline, HAC
  corrected (`evaluation/diebold_mariano.py`).
- **Deflated Sharpe** — the median-seed per-observation Sharpe deflated by the honest
  multiplicity `n_trials = #seeds × #HP configs` (`evaluation/dsr.py`).
- **PBO** — the CSCV probability of backtest overfitting (`evaluation/pbo.py`).

`evaluation/verdict.py::derive_verdict` is a **pure** function of these scalars. It
returns `rl_beats_baselines = True` only if **all four** gates hold (DM-favours AND
`deflated_sharpe > 1 − alpha` AND `seed_sharpe_lo > 0` AND `pbo < 0.5`); otherwise the
verdict is `no_significant_difference`. The DSR gate is a **confidence** threshold, not a
`> 0` test (ADR-0005) — a probability can be positive yet far below 0.95.

The committed run lands at `rl_beats_baselines = false`: only PBO clears its gate; the DM
is insignificant and wrong-signed, the DSR is ~0, and the across-seed lower bound is
below zero. See the README Validation table for the exact numbers.

### Anti-vacuous-null sanity

A null is only honest if the machinery *can* detect skill when it exists. The
`learnable_edge` synthetic fixture seeds one asset with a persistent risk-adjusted
premium; a regression asserts the allocator tilts toward it and beats 1/N. On
`synthetic` / `pure_noise` it does not — that contrast is the proof the null is honest,
not an artifact of broken plumbing.

## 7. Offline train → ONNX serve

`agents/ppo.py` (the `[train]` extra) wraps SB3 PPO, extracts the policy MLP, and exports
**only** that network to ONNX, validating it against the torch forward pass to **1e-4** at
export time. `agents/onnx_policy.py` serves the committed `artifacts/policy.onnx` via
`onnxruntime` and projects the scores onto the simplex — it never imports torch
(ADR-0006). `train.py` orchestrates the whole offline pipeline behind an injectable
`trainer` seam, so the orchestration is covered torch-free by injecting a numpy+onnx
trainer; `serve.py` only ever reads the committed artifacts and re-derives the verdict
from the live walk-forward DM plus the committed DSR / seed-lo / PBO.

## 8. Data

`data/synthetic.py` generates a `factor_regime` panel (K asset classes via a factor model
with regime-switching correlations + idiosyncratic noise, seeded by `_rng`) where by
construction no allocation beats 1/N net of costs; plus the `learnable_edge` sanity
variant and a `pure_noise` strict null. `data/loaders.py` defaults to synthetic and only
lazily tries the optional real providers. Polygon ETFs use the existing key; EODHD is the
optional paid cross-asset path. The deployed default requires **no key**.

## 9. Quality gates

- `ruff check` + `ruff format --check` — clean.
- `mypy src` — strict, clean.
- `pytest` with `fail_under = 85` coverage — the partitions are
  `tests/{unit,parity,property,regression,integration}/`.
- A `no-ai-attribution` CI guard and import-purity smoke test.
