"""Stop-criterion checker for the closed-loop AL.

See DESIGN.md §3.5 (S.1-S.3) and §4.8.

Stop when ANY:
  S.1  Budget exhausted (DFT calls or wallclock).
  S.2  Validation MAE plateau 2 consecutive rounds (rel. change < 5%).
  S.3  Tier-1 physics check fails (escalation, not graceful stop).
"""

from __future__ import annotations

from dataclasses import dataclass

from anvil.state import StateCheckpoint


@dataclass
class StopCriteria:
    budget_dft_calls: int
    max_walltime_h: int = 72
    plateau_rounds: int = 2
    plateau_threshold: float = 0.05    # relative change in val MAE


def should_stop(ckpt: StateCheckpoint, criteria: StopCriteria) -> tuple[bool, str]:
    """Return (stop, reason) given current checkpoint and criteria.

    Reasons (priority order):
      'budget'       — dft_calls_used >= budget_dft_calls
      'physics_fail' — Tier-1 check failed in last RoundMonitor entry
      'plateau'      — val MAE plateau over plateau_rounds rounds
      ''             — continue
    """
    if ckpt.dft_calls_used >= criteria.budget_dft_calls:
        return True, "budget"

    if ckpt.monitor:
        last = ckpt.monitor[-1]
        if not last.tier1_passed:
            return True, "physics_fail"

        if len(ckpt.monitor) >= criteria.plateau_rounds + 1:
            recent = ckpt.monitor[-(criteria.plateau_rounds + 1):]
            e_vals = [m.val_e_mae for m in recent]
            denom = max(1e-9, abs(e_vals[0]))
            rel_change = max(
                abs(e_vals[i] - e_vals[i - 1]) / denom
                for i in range(1, len(e_vals))
            )
            if rel_change < criteria.plateau_threshold:
                return True, "plateau"

    return False, ""
