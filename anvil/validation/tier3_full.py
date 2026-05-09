"""Tier 3 — full broad-indtest. Opt-in, ~150 DFT calls.

See DESIGN.md §5.3.

Adds vs Tier 2:
  - Hot MD probes at every T in regime (not just hottest).
  - 5 strain factors instead of 2.
  - Transition-region probes if PALLAS pathway provided.
  - Full phonon dispersion check on at least one phase.
"""

from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator


def build_tier3_indtest(
    seed_phases: dict[str, str],
    pressures_gpa: list[float],
    temperatures_k: list[float],
    foundation_calc: Calculator,
    pallas_pathway_xyz: str | None = None,
) -> list[Atoms]:
    raise NotImplementedError("week 4 (opt-in)")
