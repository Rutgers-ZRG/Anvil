"""Unit tests for AL stop criteria."""

from __future__ import annotations

from anvil.al.stop import StopCriteria, should_stop
from anvil.state import AnvilState, RoundMonitor, StateCheckpoint


def _ckpt(**kw) -> StateCheckpoint:
    base = dict(run_id="r", system="x", state=AnvilState.ROUND_VALIDATED)
    base.update(kw)
    return StateCheckpoint(**base)


def test_budget_exhausted() -> None:
    c = _ckpt(dft_calls_used=900)
    crit = StopCriteria(budget_dft_calls=800)
    stop, reason = should_stop(c, crit)
    assert stop and reason == "budget"


def test_physics_fail() -> None:
    c = _ckpt(dft_calls_used=200, monitor=[
        RoundMonitor(round=0, pool_size=180, dft_calls_used_cumulative=200,
                     tier1_passed=False, tier1_failures=["phonon"]),
    ])
    crit = StopCriteria(budget_dft_calls=800)
    stop, reason = should_stop(c, crit)
    assert stop and reason == "physics_fail"


def test_plateau_triggers() -> None:
    c = _ckpt(dft_calls_used=400, monitor=[
        RoundMonitor(round=0, pool_size=100, dft_calls_used_cumulative=100, val_e_mae=10.0),
        RoundMonitor(round=1, pool_size=200, dft_calls_used_cumulative=200, val_e_mae=10.05),
        RoundMonitor(round=2, pool_size=300, dft_calls_used_cumulative=300, val_e_mae=10.10),
    ])
    crit = StopCriteria(budget_dft_calls=800, plateau_rounds=2,
                         plateau_threshold=0.02)
    stop, reason = should_stop(c, crit)
    assert stop and reason == "plateau"


def test_no_stop_yet() -> None:
    c = _ckpt(dft_calls_used=200, monitor=[
        RoundMonitor(round=0, pool_size=100, dft_calls_used_cumulative=100, val_e_mae=20.0),
        RoundMonitor(round=1, pool_size=200, dft_calls_used_cumulative=200, val_e_mae=10.0),
    ])
    crit = StopCriteria(budget_dft_calls=800, plateau_rounds=2, plateau_threshold=0.05)
    stop, reason = should_stop(c, crit)
    assert not stop and reason == ""
