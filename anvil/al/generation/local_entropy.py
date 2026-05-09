"""Local entropy bias (Karabin & Perez 2020) — thin wrapper over reformpy.

See DESIGN.md §3.2 Pool A. Imported, not vendored.
"""

from __future__ import annotations

from ase.calculators.calculator import Calculator


def make_local_entropy_calc(
    base_calc: Calculator,
    k_factor: float = 1.0,
    cutoff: float = 4.0,
    natx: int | None = None,
) -> Calculator:
    """Return reformpy.EntropyMaximizingCalculator wrapping base_calc.

    Forwards the standard reformpy parameters; we don't customize.
    """
    raise NotImplementedError(
        "week 1: from reformpy import EntropyMaximizingCalculator; "
        "return EntropyMaximizingCalculator(calculator=base_calc, "
        "k_factor=k_factor, cutoff=cutoff, natx=natx)"
    )
