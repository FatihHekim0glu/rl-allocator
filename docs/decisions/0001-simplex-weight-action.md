# ADR-0001 — Simplex weight action

- Status: Accepted
- Context: generalizing the single-position rl-trader env to a multi-asset allocator.

## Context

The reused rl-trader environment exposes a scalar position action. A portfolio allocator
must instead emit a **vector** of target weights across `n_assets`, and a portfolio
weight vector is only meaningful if it is a valid simplex: long-only (`w ≥ 0`) and fully
invested (`Σw = 1`). Emitting raw weights and hoping the agent learns the constraint is
fragile — a single bar with negative or unnormalized weights silently corrupts every
downstream metric (Sharpe, drawdown, turnover) and the baselines comparison.

## Decision

The action is a **raw per-asset score vector** `(n_assets,)` that is **projected onto the
long-only simplex** before it is ever used. `env/portfolio_env.py` projects via
`rlallocator._validation.project_to_simplex`, so the held weights satisfy `Σw = 1` and
`w ≥ 0` **every bar by construction**. The policy network outputs scores; the projection
— not the network — guarantees validity. The same projection is applied on the serve
path so the ONNX policy's scores become a valid simplex identically.

## Consequences

- The weight vector is a valid simplex on every bar without relying on learned behavior;
  a Hypothesis property test (`tests/property/test_env_properties.py`) asserts this for
  arbitrary score vectors.
- The action space is unconstrained scores (easy for PPO to optimize over) while the held
  state is always a feasible portfolio.
- Long/short budgets are a documented future extension; the shipped allocator is
  long-only.
- The verdict and all OOS metrics operate on guaranteed-valid weight paths, so the
  honest-NULL result cannot be an artifact of malformed weights.
