# ADR-0002 — Causal next-bar reward

- Status: Accepted
- Context: leakage is the most common silent failure in backtested RL.

## Context

The single most common way a financial RL backtest produces a fake positive result is
look-ahead: the reward at bar `t` is computed from information that would not be available
when the weights are chosen, or the observation at `t` peeks at future bars. Either leak
inflates Sharpe and would make the honest-NULL deliverable a lie.

## Decision

The reward is **strictly next-bar and causal**. In `env/portfolio_env.py`:

- the observation at bar `t` uses only data `≤ t` (the look-back window of per-asset
  returns/features plus the current weights);
- the weights `w_t` chosen at `t` earn the **next** bar's asset returns; the reward is

  ```
  reward_t = w_t · r_{t -> t+1} - (cost_bps / 1e4) * ‖w_t - w_{t-1}‖_1
  ```

- the env requires at least two bars so a causal `r_{t -> t+1}` step exists, and the
  baselines (Markowitz / risk-parity) estimate covariance on the **train fold only**.

## Consequences

- A Hypothesis property test perturbs future bars and asserts the obs and reward at `t`
  are invariant (future-perturbation invariance) — the leak tripwire.
- The turnover term makes costs causal too: rebalancing into `w_t` is charged at `t`, not
  retroactively.
- Combined with the parity oracle (ADR-0003), causality is enforced from two angles: the
  env defines a causal reward, and the vectorized backtester is proven identical to it,
  so neither path can silently leak.
