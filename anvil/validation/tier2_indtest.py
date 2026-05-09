"""Tier 2 — mini broad-indtest. Default validation, ~50 DFT calls.

See DESIGN.md §5.2.

Built once at bootstrap from foundation MLIP, never enters training.
Construction logic ports build_broad_indtest.py from mlip-active-learn.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator


def build_tier2_indtest(
    seed_phases: dict[str, str],
    pressures_gpa: list[float],
    temperatures_k: list[float],
    foundation_calc: Calculator,
    *,
    strain_factors: tuple[float, float] = (0.92, 1.08),
    n_hot_snapshots: int = 5,
) -> list[Atoms]:
    """Construct the Tier-2 mini broad-indtest seed list.

    Per (phase × P): 2 strain probes (no thermal noise → high stress).
    Per phase at hottest T: 5 hot-MD snaps (high force).

    These are SEEDS, not labels — DFT labeling happens in parallel with the
    bootstrap pool labeling.
    """
    raise NotImplementedError(
        "week 1: ports build_broad_indtest.py from mlip-active-learn"
    )


def evaluate_tier2(
    ensemble_calc: Calculator,
    indtest_xyz: str | Path,
    thresholds: dict,
) -> dict:
    """Evaluate ensemble on the labeled Tier-2 set.

    Returns {'E_MAE': ..., 'F_MAE': ..., 'S_MAE': ..., 'passed': bool,
             'subset_metrics': {'strain': {...}, 'hot': {...}}}.
    """
    raise NotImplementedError("week 1")
