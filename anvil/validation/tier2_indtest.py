"""Tier 2 — mini broad-indtest. Default validation, ~50 DFT calls.

See DESIGN.md §5.2.

Built once at bootstrap, never enters training. Construction logic ports
build_broad_indtest.py from mlip-active-learn but in v1 piece-3 minimum
generates ONLY strain probes (no hot-MD, since hot MD needs the foundation
MLIP wired). Strain factors are wider than Pool C anchor's so the validation
set is genuinely independent.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

from ase import Atoms

from anvil.al.generation.anchor import _apply_isotropic_strain, _read_seed


# Wider strain factors than Pool C anchor — Tier-2 should probe a broader
# region than training data. Matches build_broad_indtest.py defaults.
TIER2_STRAIN_FACTORS: tuple[float, float] = (0.92, 1.08)


def build_tier2_indtest_seeds(
    seed_phases: dict[str, str | Path],
    pressures_gpa: list[float],
    temperatures_k: list[float] | None = None,
    *,
    strain_factors: tuple[float, float] = TIER2_STRAIN_FACTORS,
    n_hot_per_phase: int = 0,             # v1 piece-3: 0 (no foundation MLIP)
) -> list[Atoms]:
    """Construct the Tier-2 mini broad-indtest seeds.

    v1 piece-3 minimum: strain probes only, 2 factors × 5 P × N_phases.
    For carbon (5 phases × 5 P × 2 factors) = 50 cells. Matches DESIGN.md §5.2.

    n_hot_per_phase reserved for week-2 foundation MLIP integration.
    """
    out: list[Atoms] = []
    for phase, seed_path in seed_phases.items():
        seed = _read_seed(seed_path)
        for P in pressures_gpa:
            for factor in strain_factors:
                a = _apply_isotropic_strain(seed, factor)
                a.info["pool"] = "validation"
                a.info["phase"] = phase
                a.info["pressure_gpa"] = float(P)
                a.info["temperature_k"] = 0
                a.info["sample_kind"] = f"strain_{factor:.2f}"
                out.append(a)

    if n_hot_per_phase > 0:
        raise NotImplementedError(
            "Hot MD probes require foundation MLIP integration (week 2)."
        )
    return out
