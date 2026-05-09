"""Pool B — Reform-relax (CRISP-style, FP-targeted toward seed phases).

See DESIGN.md §3.3.

Drives initial structures toward ordered phases via reformpy.Reform_Calculator
+ foundation MLIP base. Default mode (a) targets nearest seed-phase FP for
stability; opt-in mode (b) targets farthest seed-phase FP for phase
discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


@dataclass
class ReformRelaxConfig:
    mode: Literal["stability", "discovery"] = "stability"
    initial_sources: list[str] = field(
        default_factory=lambda: ["perturbed_seed", "pyxtal", "pool_a_low_f"]
    )
    pyxtal_n_per_round: int = 20
    perturb_rattle: float = 0.3      # Å
    perturb_cell: float = 0.05       # fractional
    relax_max_steps: int = 200
    relax_fmax: float = 0.05         # eV/Å
    sample_intermediates: int = 2
    spglib_filter: bool = True
    pallas_pathway_xyz: Optional[str] = None


def generate_pool_b(
    config: ReformRelaxConfig,
    base_calc: Calculator,
    *,
    seed_phase_fps: dict[str, np.ndarray],   # phase_name -> mean FP at relaxed
    n_target: int,
    pool_a_low_f_atoms: Optional[list[Atoms]] = None,
    composition: dict[str, int] | None = None,    # for pyXtal random gen
) -> list[Atoms]:
    """Generate Pool B candidates.

    Steps:
    1. Build initial-structure mix (perturbed seed / pyxtal / pool_a_low_f).
    2. For each, select FP target (nearest in mode=stability, farthest in
       mode=discovery) based on candidate's current pressure or composition.
    3. Run reformpy.Reform_Calculator + base_calc relaxation (BFGS/FIRE).
    4. Sample final relaxed cell + 1-2 intermediates.
    5. Optional spglib filter (drop P1, keep ≥ Pmnm).
    6. FP-distance dedupe within Pool B.

    Returns up to n_target ordered crystal-like structures.
    """
    raise NotImplementedError("week 2: see DESIGN.md §3.3")


def select_fp_target(
    current_fp: np.ndarray,
    seed_phase_fps: dict[str, np.ndarray],
    mode: Literal["stability", "discovery"],
) -> tuple[str, np.ndarray]:
    """Return (phase_name, target_fp) under the chosen mode.

    stability: nearest FP   (drives toward closest known phase)
    discovery: farthest FP  (drives toward novel basins)
    """
    raise NotImplementedError("week 2")
