"""Unit tests for the Typer CLI (commands invoked via their plain function bodies)."""

from __future__ import annotations

import importlib.util

import pytest

from rlallocator.cli import backtest, build_app, compare, main, train

#: The offline ``[train]`` extra availability; the ``train`` command's
#: trainer-unavailable behaviour only holds without it (else it would train).
_HAS_TRAIN: bool = bool(importlib.util.find_spec("torch"))


@pytest.mark.unit
def test_build_app_registers_three_commands() -> None:
    """build_app returns a Typer app with train / backtest / compare registered."""
    app = build_app()
    names = {cmd.name for cmd in app.registered_commands}
    assert {"train", "backtest", "compare"} <= names


@pytest.mark.unit
def test_backtest_command_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    """The backtest command runs the live baselines and prints the OOS table (exit 0)."""
    code = backtest(n_assets=4, n_seeds=3, cost_bps=10.0, lookback=16, seed=7)
    assert code == 0
    out = capsys.readouterr().out
    assert "1/N" in out
    assert "Markowitz" in out
    assert "risk-parity" in out


@pytest.mark.unit
def test_compare_command_prints_verdict(capsys: pytest.CaptureFixture[str]) -> None:
    """The compare command prints the honest rl_beats_baselines verdict (NO by default)."""
    code = compare(n_assets=4, n_seeds=3, cost_bps=10.0, lookback=16, seed=7)
    assert code == 0
    out = capsys.readouterr().out
    assert "rl_beats_baselines: NO" in out
    assert "PBO" in out


@pytest.mark.unit
@pytest.mark.skipif(_HAS_TRAIN, reason="[train] extra installed: the train command would proceed")
def test_train_command_unavailable_without_train_extra(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without the [train] extra, the train command surfaces it cleanly (exit 1).

    The lean serve / CI env has no torch / sb3 / gymnasium, so the default SB3 trainer is
    unavailable; the CLI catches the NotImplementedError and exits non-zero with a clean
    message rather than crashing.
    """
    code = train(n_assets=4, n_seeds=2, lookback=16, cost_bps=10.0, seed=7)
    assert code == 1
    assert "train unavailable" in capsys.readouterr().out


@pytest.mark.unit
def test_main_builds_and_runs_with_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """main builds the app and runs it; --help exits cleanly (SystemExit 0)."""
    monkeypatch.setattr("sys.argv", ["rl-allocator", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
