# ADR-0003 — Vectorized ⇄ stepwise parity oracle

- Status: Accepted
- Context: the served metrics use the fast vectorized backtester, not the env rollout.

## Context

The env (`env/portfolio_env.py`) defines the causal reward one bar at a time, but
evaluating every seed across the full panel that way is slow. The serve and evaluation
paths use a **vectorized** equity-curve evaluator (`env/backtester.py`). Two
implementations of the same dynamics will drift unless something pins them together — and
a subtle drift (an off-by-one in the return alignment, say) is exactly how look-ahead
sneaks back in after ADR-0002.

## Decision

Ship a **parity oracle** (`env/parity.py`) that asserts the vectorized backtester equals
the step-by-step env rollout to **1e-10** for arbitrary weight paths, and ship a
deliberately-leaky negative control (`leaky_backtest`) that the oracle is asserted to
**catch**.

- `check_parity` / `assert_parity` compare the two equity curves bar-by-bar at 1e-10.
- `leaky_backtest` is a variant that intentionally uses `r_{t-1 -> t}` (look-ahead);
  `tests/regression/test_leaky_negative_control.py` asserts the oracle flags it as a
  parity violation.

## Consequences

- The fast path used in production is provably the same dynamics as the causal env, to
  numerical precision.
- The negative control proves the oracle has teeth: it is not a tautology that always
  passes. If the oracle could not catch a known leak, the 1e-10 pass would be
  meaningless.
- Any future change to either backtester that introduces look-ahead breaks the parity
  test, so the leak is caught in CI rather than in the headline metric.
