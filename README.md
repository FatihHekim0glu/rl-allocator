# rl-allocator

**A leakage-free, overfit-aware multi-asset RL portfolio allocator — an honest NULL result.**

Train a PPO agent to allocate across a multi-asset basket (a simplex of portfolio
weights) in a cost-aware portfolio environment, evaluate it out-of-sample inside a
**purged walk-forward** against **equal-weight (1/N)**, **Markowitz mean-variance**,
and **risk-parity** baselines — with a **vectorized-vs-stepwise parity oracle**, a
**seed-lottery** overfit check, the **Deflated Sharpe Ratio**, and the **Probability of
Backtest Overfitting**.

## Honest headline (the deliverable)

> A PPO multi-asset portfolio allocator does **NOT** reliably beat equal-weight /
> Markowitz / risk-parity out-of-sample after turnover costs. Across training seeds the
> OOS Sharpe is dispersed around (and statistically indistinguishable from) the
> baselines after a Deflated-Sharpe correction + a Probability-of-Backtest-Overfitting
> check — the apparent skill is mostly training-path overfit (the seed lottery on the
> largest search surface).

The deliverable is the rigorous, leakage-free, parity-checked, overfit-aware multi-asset
RL backtest — **not a profit claim**. Execution is **SIMULATED** (turnover costs), never
a live broker. The shipped default runs on a **synthetic** multi-asset factor-regime
panel + a committed offline-trained ONNX policy; real cross-asset data (EODHD / Polygon
ETFs) is the **optional offline CLI path** that needs no key for the deployed tool.

## Validation — the committed result

These are the **actual numbers** from the committed offline run
(`src/rlallocator/artifacts/metrics.json`): the median-seed PPO allocator scored on the
concatenated purged walk-forward OOS folds of the synthetic factor-regime panel
(`n_assets = 6`, `n_seeds = 5`, `cost_bps = 10.0`), net of turnover costs.

| Metric | Value | Reading |
| --- | ---: | --- |
| RL median OOS Sharpe | **−1.79** | The median-seed PPO allocator, net of costs. |
| 1/N (equal-weight) OOS Sharpe | −0.92 | — |
| Markowitz mean-variance OOS Sharpe | −1.07 | — |
| Risk-parity OOS Sharpe | −0.97 | — |
| **Best baseline** | **`equal_weight`** (−0.92) | The bar the RL agent must clear. |
| Across-seed Sharpe band `[lo, hi]` | **[−2.13, −1.28]** | The seed lottery — the lower bound is **< 0**, so gate (3) fails. |
| Diebold-Mariano p vs. best | **0.125** | `> alpha` (0.05) and the DM statistic is negative, so gate (1) fails. |
| Deflated Sharpe (n_trials = 5) | **0.00** | Far below the `1 − alpha` = 0.95 confidence level, so gate (2) fails. |
| Probability of Backtest Overfitting (CSCV) | **0.214** | `< 0.5`, so gate (4) is the only one that holds. |
| Median-seed max drawdown | −0.706 | Worst peak-to-trough on the OOS equity. |
| Median-seed total one-way turnover | 999.8 | Cumulative `Σ‖Δwₜ‖₁` over the OOS path (heavy daily rebalancing). |
| `n_effective_trials` (#seeds × #HP) | 5 | The honest multiplicity used by the DSR. |
| Data source | `synthetic` (factor_regime) | The shipped default; no key required. |
| **`rl_beats_baselines`** | **`false`** | The pure verdict — the honest NULL. |

**The verdict is `false` because three of the four gates fail** (DM is insignificant and
the wrong sign, the Deflated Sharpe is far below 0.95, and the across-seed lower bound
straddles below zero); only PBO `< 0.5` holds. The agent is *worse* than 1/N here, and
even the best the seeds produce is statistically indistinguishable from the baselines.

## The pure verdict — `rl_beats_baselines`

`rlallocator.evaluation.verdict.derive_verdict` is a **pure function** of the inference
outputs. It reads `True` **only if ALL FOUR** gates hold for the median-seed allocator
vs. the **best** of the three baselines, net of costs:

1. **Diebold-Mariano** on the per-bar net-return differential is significant **and**
   signed in the RL agent's favour (`dm_pvalue < alpha` **and** `dm_statistic > 0`);
2. the **Deflated Sharpe** (honest `n_trials = #seeds × #HP configs`) clears the
   `1 − alpha` **confidence** level (`deflated_sharpe > 0.95`) — the DSR is a
   *probability*, so the gate is a confidence threshold, **never** `> 0`;
3. the **across-seed Sharpe lower bound** is strictly positive (`seed_sharpe_lo > 0` —
   the seed-lottery dispersion does not straddle zero);
4. the **Probability of Backtest Overfitting** (CSCV) is below one-half (`pbo < 0.5`).

If any gate fails, the verdict is `no_significant_difference` — the documented,
leakage-free, honest-NULL outcome.

## Correctness gates (what is asserted, and to what tolerance)

The honesty stack is the deliverable, so every claim is pinned by a test. Run them with
`pytest` (see [Reproduce](#reproduce)); the partitions live under
`tests/{parity,property,regression,integration,unit}/`.

| Gate | Mechanism | Tolerance / criterion |
| --- | --- | --- |
| Vectorized ⇄ stepwise parity | `env/parity.py` checks the vectorized multi-asset backtester equals the step-by-step `PortfolioEnv` rollout for arbitrary weight paths. | **1e-10** (exact) |
| Leaky negative control | `env/parity.py::leaky_backtest` is a deliberately look-ahead variant; the oracle is asserted to **catch** it. | parity assertion **must fail** |
| ONNX ⇄ torch policy | `agents/ppo.py::export_onnx` validates the exported policy MLP against the torch forward pass at export time. | **1e-4** (relative) |
| Deflated Sharpe | `evaluation/dsr.py` reproduced against a reference implementation. | **1e-10** |
| DSR confidence gate | A positive-but-sub-0.95 DSR **fails** the verdict (the DSR is a probability, not a `> 0` test). | `deflated_sharpe > 1 − alpha` |
| Diebold-Mariano | `evaluation/diebold_mariano.py` HAC-corrected DM statistic + sign. | hand-reference + degeneracy cases |
| PBO / CSCV | `evaluation/pbo.py` Combinatorially-Symmetric Cross-Validation. | vs. reference |
| Causal obs / reward | Weights at `t` earn `t → t+1`; obs at `t` uses only data `≤ t`. | future-perturbation **invariance** (Hypothesis) |
| Valid simplex every bar | The held weight vector sums to 1 and is long-only `≥ 0`. | property test (Hypothesis) |
| Cost monotonicity | More turnover ⇒ lower net return. | property test (Hypothesis) |
| Walk-forward in served path | `serve.run_allocation` computes the headline OOS metrics from the **concatenated purged folds** (baselines AND the committed ONNX policy), not the full sample. | regression asserts the served path **calls** the walk-forward fn |
| Train-only covariance | Markowitz / risk-parity estimate covariance on the **train fold only**. | no look-ahead (unit) |
| `learnable_edge` sanity | On the premium-asset fixture the allocator **does** tilt toward the premium and beat 1/N (so the null is honest, not vacuous). | regression (pinned) |
| Honest null | On synthetic / `pure_noise` the agent does **not** beat the baselines after costs + DSR + PBO + seed dispersion; deterministic across `PYTHONHASHSEED`. | regression (pinned) |

## Install

```bash
uv venv && uv pip install -e '.[data,serve,viz,dev]'   # lean: onnxruntime, NO torch/sb3
uv pip install -e '.[train]'                             # offline training only (torch + SB3)
```

The lean serve path imports **onnxruntime only** — never torch / sb3 / gymnasium.
`import rlallocator` is **import-pure** (zero heavy deps at module load).

## Reproduce

```bash
# 1. Lean serve environment (no torch / sb3 / gymnasium)
uv venv && uv pip install -e '.[data,serve,viz,dev]'

# 2. Run the served comparison on the synthetic panel (live baselines + committed ONNX
#    policy, both scored on the purged walk-forward), and print the pure verdict.
rl-allocator compare

# 3. Inspect the committed honest-NULL metrics directly.
rl-allocator backtest      # live baselines (+ committed ONNX policy) on a synthetic panel

# 4. Quality gates — all must pass.
ruff check src tests
ruff format --check src tests
mypy src                   # strict
pytest -q                  # coverage gate fail_under = 85
```

### Offline training (the `[train]` extra)

Training is **offline** and produces the committed artifacts; the request/serve path
never trains.

```bash
uv pip install -e '.[train]'                  # torch (CPU) + stable-baselines3 + gymnasium
rl-allocator train                            # the full offline pipeline
python scripts/train_committed_policy.py      # reproduces artifacts/{policy.onnx, metrics.json}
```

The pipeline: synthetic panel → **purged/embargoed walk-forward** → PPO across N seeds →
export each policy MLP to ONNX (validated **1e-4** vs torch) → score each FROZEN policy
on the concatenated purged OOS folds → seed lottery + DM-vs-best + Deflated Sharpe +
CSCV PBO → the PURE `rl_beats_baselines` verdict → committed `< 10 MB` `policy.onnx` +
`metrics.json` + a `RunManifest`.

### Real cross-asset data (optional, offline only)

The deployed default is synthetic and needs **no key**. Real multi-asset data is an
**optional offline CLI path**:

- **Polygon ETFs** — uses the existing key. The provider resolves the key from an
  explicit argument, the `POLYGON_API_KEY` environment variable, or a repo-root `.env`:

  ```bash
  export POLYGON_API_KEY=...      # or put it in .env
  rl-allocator backtest --data-source polygon
  ```

- **EODHD** — broad cross-asset coverage behind a **paid** EODHD key the deployed tool
  does **not** require (`--data-source eodhd`). The reader is the optional offline path;
  absent a key it falls through to the deterministic synthetic panel.

## Limitations

- **Execution is SIMULATED.** Costs are a turnover (L1) penalty only — there is no live
  broker, and no market-impact, slippage, borrow, financing, or liquidity model.
- **The deployed default is synthetic-trained.** The committed policy + metrics come from
  the synthetic factor-regime panel, which is constructed so that no allocation beats 1/N
  net of costs (the honest null holds by design). Real cross-asset data is **optional**
  via **EODHD (paid)** or **Polygon ETFs** (the offline CLI path) — the deployed tool
  needs no key.
- **The result is a seed lottery** on the largest search surface (a weight simplex). The
  headline is the across-seed dispersion + DSR + PBO, **never** a single-seed equity
  curve, and the verdict is gated accordingly.
- **Rebalancing is idealized** — no fractional-share, lot-size, or liquidity constraints;
  the committed run rebalances every bar, hence the large cumulative turnover.
- **Synthetic data is not a market claim.** It exercises the leakage / parity / overfit
  machinery; it is not evidence about any real asset class.

## Design & decisions

- [`docs/DESIGN.md`](docs/DESIGN.md) — the system design: the causal env, the parity
  oracle, the walk-forward-in-served-path wiring, the offline-train → ONNX-serve split,
  and the honesty stack.
- [`docs/decisions/`](docs/decisions) — the Architecture Decision Records:
  - [ADR-0001](docs/decisions/0001-simplex-weight-action.md) — simplex weight action.
  - [ADR-0002](docs/decisions/0002-causal-next-bar-reward.md) — causal next-bar reward.
  - [ADR-0003](docs/decisions/0003-parity-oracle.md) — the vectorized↔stepwise parity oracle.
  - [ADR-0004](docs/decisions/0004-walk-forward-served-path.md) — walk-forward wired into the served path.
  - [ADR-0005](docs/decisions/0005-dsr-confidence-gate.md) — the DSR confidence gate.
  - [ADR-0006](docs/decisions/0006-onnx-policy-serve.md) — ONNX-policy serve (no torch at request time).

## References

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.
- Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest
  Overfitting* (CSCV).
- Markowitz (1952), *Portfolio Selection*.
- de Prado (2018), *Advances in Financial Machine Learning* (purged CV, embargo).

## License

MIT — see [LICENSE](LICENSE).
