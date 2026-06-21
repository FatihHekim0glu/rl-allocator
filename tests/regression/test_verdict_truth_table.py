"""Regression: the PURE rl_beats_baselines verdict truth table + the DSR confidence gate."""

from __future__ import annotations

import pytest

from rlallocator._exceptions import ValidationError
from rlallocator.evaluation.verdict import Verdict, derive_verdict

# A passing configuration: DM significant & positive, DSR > 0.95, seed-lo > 0, PBO < 0.5.
_PASS = {
    "dm_statistic": 3.0,
    "dm_pvalue": 0.001,
    "deflated_sharpe": 0.99,
    "seed_sharpe_lo": 0.2,
    "pbo": 0.1,
    "n_effective_trials": 5,
}


@pytest.mark.regression
def test_all_four_gates_pass_yields_rl_beats() -> None:
    """When all four gates pass, the verdict is RL_BEATS_BASELINES."""
    result = derive_verdict(**_PASS)
    assert result.rl_beats_baselines is True
    assert result.verdict is Verdict.RL_BEATS_BASELINES


@pytest.mark.regression
@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"dm_pvalue": 0.5}, "DM insignificant"),
        ({"dm_statistic": -3.0}, "DM signed against RL"),
        ({"deflated_sharpe": 0.80}, "DSR below the 1-alpha confidence level"),
        ({"seed_sharpe_lo": -0.01}, "seed lower bound straddles zero"),
        ({"pbo": 0.6}, "PBO at/above one-half"),
    ],
)
def test_any_single_gate_failure_yields_null(override: dict[str, float], reason: str) -> None:
    """Failing ANY single gate flips the verdict to NO_SIGNIFICANT_DIFFERENCE."""
    cfg = {**_PASS, **override}
    result = derive_verdict(**cfg)  # type: ignore[arg-type]
    assert result.rl_beats_baselines is False, reason
    assert result.verdict is Verdict.NO_SIGNIFICANT_DIFFERENCE


@pytest.mark.regression
def test_positive_but_sub_095_dsr_fails() -> None:
    """A positive-but-sub-0.95 DSR FAILS the gate (DSR is a probability, not > 0)."""
    cfg = {**_PASS, "deflated_sharpe": 0.50}
    assert derive_verdict(**cfg).rl_beats_baselines is False  # type: ignore[arg-type]
    # And a DSR just below 0.95 still fails; just above passes (with the other gates ok).
    assert derive_verdict(**{**_PASS, "deflated_sharpe": 0.9499}).rl_beats_baselines is False  # type: ignore[arg-type]
    assert derive_verdict(**{**_PASS, "deflated_sharpe": 0.9501}).rl_beats_baselines is True  # type: ignore[arg-type]


@pytest.mark.regression
def test_verdict_validates_pbo_and_pvalue_ranges() -> None:
    """The verdict rejects out-of-range PBO / p-values."""
    with pytest.raises(ValidationError):
        derive_verdict(**{**_PASS, "pbo": 1.5})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        derive_verdict(**{**_PASS, "dm_pvalue": 2.0})  # type: ignore[arg-type]
