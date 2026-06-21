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

## Validation table

| Guard | Mechanism |
| --- | --- |
| Causal reward | Weights set at `t` earn `t → t+1` returns; obs at `t` uses only data `≤ t` (future-perturbation-invariance property test). |
| Parity oracle | Vectorized multi-asset backtester == step-by-step env rollout to **1e-10** for arbitrary weight paths, plus a deliberately-leaky negative control the oracle catches. |
| Purged walk-forward | `serve.run_allocation` computes the headline OOS metrics from the **concatenated purged folds** (purge ≥ 1 + embargo = 1), not the full sample; a regression asserts the served path calls the walk-forward fn. |
| Train-only covariance | Markowitz / risk-parity estimate covariance on the **train fold only** (no look-ahead). |
| Valid simplex | The weight vector is a valid long-only simplex **every bar** (a property test). |
| Seed lottery | N independent training seeds; the verdict requires the across-seed Sharpe **lower bound > 0**. |
| Deflated Sharpe | Gated at the `1 − alpha` **confidence** level (a positive-but-sub-0.95 DSR fails). |
| PBO | CSCV Probability of Backtest Overfitting; the verdict requires `pbo < 0.5`. |
| `learnable_edge` sanity | On the premium-asset fixture the allocator **does** tilt toward the premium and beat 1/N (so the null is honest, not vacuous). |

## Install

```bash
uv venv && uv pip install -e '.[data,viz,dev]'   # lean: NO torch/sb3/gymnasium
uv pip install -e '.[train]'                       # offline training only (torch + SB3)
```

The lean serve path imports **onnxruntime only** — never torch / sb3 / gymnasium.
`import rlallocator` is **import-pure** (zero heavy deps at module load).

## CLI

```bash
rl-allocator backtest   # live baselines (+ committed ONNX policy) on a synthetic panel
rl-allocator compare    # the full RL-vs-baselines comparison + the pure verdict
rl-allocator train      # the OFFLINE pipeline (the [train] extra)
```

## Limitations

- **Execution is SIMULATED** (turnover costs only); there is no live broker, no
  market-impact / borrow / financing model.
- The deployed default is **synthetic-trained**; real cross-asset data is optional via
  EODHD / Polygon (the offline CLI path).
- Rebalancing is **idealized** (no fractional-share / liquidity constraints).
- The result is a **seed lottery** on the largest search surface — a single-seed equity
  curve is never the headline.

## References

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.
- Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest
  Overfitting* (CSCV).
- Markowitz (1952), *Portfolio Selection*.
- de Prado (2018), *Advances in Financial Machine Learning* (purged CV, embargo).

## License

MIT — see [LICENSE](LICENSE).
