"""Convergence + sanity checks on VASP output before pool admission.

See DESIGN.md §4.3, §10 (NaCl B2 hot-MD timeout lesson).
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms


def is_converged(struct_dir: str | Path) -> bool:
    """True if OUTCAR contains 'reached required accuracy' or 'Total CPU time'."""
    raise NotImplementedError("week 1")


def force_sanity_check(atoms: Atoms, max_force_eV_A: float = 100.0) -> bool:
    """True if all atomic |F| < threshold. Catches broken/imploded cells."""
    raise NotImplementedError("week 1")


def stress_sign_check(atoms: Atoms, expected_convention: str = "ase") -> bool:
    """Verify stress sign convention. ASE = compression-negative.

    Lessons learned: carbon training xyz was compression-positive, mismatched
    indtest (compression-negative). Anvil tags every collected Atoms with
    stress_convention metadata.
    """
    raise NotImplementedError("week 1: see DESIGN.md §10")
