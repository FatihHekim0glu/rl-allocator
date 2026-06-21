# ADR-0005 — Deflated Sharpe confidence gate

- Status: Accepted
- Context: the DSR is a probability; treating it as a `> 0` test never binds.

## Context

The Deflated Sharpe Ratio (Bailey & López de Prado 2014) deflates an observed Sharpe by
the multiplicity of the search and returns the **probability** that the true Sharpe is
positive given how many configurations were tried. A naive verdict might gate on
`deflated_sharpe > 0`. But a probability is essentially always positive — that test would
pass for almost any backtest and never bind, defeating the entire point of the
correction. On the largest search surface in the portfolio (a weight simplex across many
seeds), the multiplicity is exactly what we must penalize.

## Decision

The verdict gates the DSR at the **`1 − alpha` confidence level**, not at zero. In
`evaluation/verdict.py::derive_verdict`:

```
dsr_ok = deflated_sharpe > (1.0 - alpha)        # e.g. > 0.95 for alpha = 0.05
```

The honest multiplicity fed to the DSR is `n_trials = #seeds × #HP configs`
(`n_effective_trials` in the metrics). The DSR uses the per-observation (un-annualized)
median-seed Sharpe.

## Consequences

- A positive-but-sub-0.95 DSR **fails** the verdict; a dedicated test pins this so the
  gate cannot silently regress to a `> 0` check.
- The committed run has `deflated_sharpe = 0.0` (n_trials = 5), far below 0.95 — gate (2)
  fails, contributing to the honest-NULL verdict.
- The gate scales with honesty: more seeds / HP configs raise `n_trials`, deflate the
  Sharpe further, and make the 0.95 bar harder to clear — exactly the discipline the
  seed-lottery deliverable demands.
