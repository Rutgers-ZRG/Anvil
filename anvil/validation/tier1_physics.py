"""Tier 1 — physics catastrophe detectors. Free, every round.

See DESIGN.md §5.1.

Checks:
  - Each seed phase relaxed with f̄: basin preserved (RMSD < 0.1 Å, no
    symmetry break).
  - E(V) at 7 volume points around equilibrium; smooth, single-minimum,
    B₀ within ±20 % of MP reference.
  - Phonon ω_min at Γ for each phase; require > -5 cm⁻¹.
  - Phase-ordering enthalpy at user pressures; require ordering matches
    DFT seeds (or foundation MLIP fallback).

If ANY fails: state machine → PAUSED_PHYSICS, user notified.
"""

from __future__ import annotations

from ase.calculators.calculator import Calculator


def run_tier1_checks(
    ensemble_calc: Calculator,
    seed_phases: dict[str, str],
    pressures_gpa: list[float],
) -> dict:
    """Return {'passed': bool, 'failures': list[str], 'details': dict}."""
    raise NotImplementedError("week 1: see DESIGN.md §5.1")
