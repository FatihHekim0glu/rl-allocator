# ADR-0004 — Walk-forward wired into the served path

- Status: Accepted
- Context: the #1 recurring bug — the entrypoint bypasses the rigorous evaluation fn.

## Context

A purged, embargoed walk-forward is the difference between an honest OOS Sharpe and a
look-ahead-contaminated one. In prior projects the *training script* used the rigorous
walk-forward, but the *served entrypoint* silently computed its headline metrics on the
**full sample** — so the number a user actually saw was not the rigorous one. This is the
single most recurring bug in the portfolio, and it is invisible unless explicitly tested.

## Decision

`serve.run_allocation` computes the headline OOS metrics — for the baselines **and** the
committed ONNX policy — from the **concatenated purged walk-forward folds**, not the full
sample.

- `walk_forward.make_folds` builds anchored, purged, embargoed folds; the purge is
  `required_purge(lookback, horizon) = lookback + horizon - 1` and the embargo defaults to
  1 (de Prado 2018).
- Each baseline estimates covariance on each fold's train block (train-only); the ONNX
  policy is served on each fold's OOS block via onnxruntime (no torch); the per-fold OOS
  blocks are concatenated and the metrics are computed on the concatenation.
- The pure verdict is re-derived from the **live** walk-forward Diebold-Mariano plus the
  committed-offline DSR / seed-lower-bound / PBO gates.

## Consequences

- The number served is the rigorous number. There is no second, looser code path.
- A regression test (`tests/regression/test_served_path_walk_forward.py`) asserts the
  served path actually **calls** the walk-forward function — the explicit guard against
  the entrypoint silently regressing to a full-sample computation.
- Absent a committed policy, the serve path degrades to a baselines-only comparison with
  the honest-NULL placeholder verdict, still over the purged folds.
