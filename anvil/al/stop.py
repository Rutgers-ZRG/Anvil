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

    `reason` is a short tag: 'budget' | 'plateau' | 'physics_fail' | None.
    """
    raise NotImplementedError("week 1: see DESIGN.md §3.5 stop criteria")
