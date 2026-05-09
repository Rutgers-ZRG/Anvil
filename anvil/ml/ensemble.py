"""K-seed Allegro fine-tune ensemble. Provides UQ via member disagreement.

See DESIGN.md §4.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


class EnsembleCalculator(Calculator):
    """ASE Calculator that wraps K member calculators.

    - get_potential_energy() → mean over K
    - get_forces() → mean over K
    - get_stress() → mean over K
    - get_uncertainty(atoms) → {'sigma_E', 'sigma_F', 'sigma_S'}
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model_paths: list[str], device: str = "cuda") -> None:
        super().__init__()
        self.model_paths = model_paths
        self.device = device

    def calculate(self, atoms: Optional[Atoms] = None,
                  properties: Optional[list[str]] = None,
                  system_changes: Optional[list[str]] = None) -> None:
        raise NotImplementedError("week 2")

    def get_uncertainty(self, atoms: Atoms) -> dict[str, float]:
        """Return per-structure σ_E, σ_F (mean), σ_S as ensemble std-deviations."""
        raise NotImplementedError("week 2")
