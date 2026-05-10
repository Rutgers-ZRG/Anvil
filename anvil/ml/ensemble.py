"""K-seed Allegro fine-tune ensemble. Provides UQ via member disagreement.

See DESIGN.md §4.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class EnsembleCalculator(Calculator):
    """ASE Calculator wrapping K NequIP member calculators.

    Properties:
      - get_potential_energy() → mean over K
      - get_forces()           → mean over K
      - get_stress()           → mean over K
      - get_uncertainty(atoms) → {'sigma_E', 'sigma_F_mean', 'sigma_F_max',
                                   'sigma_S'}

    Members are loaded lazily on first `calculate` call so the constructor
    is cheap (callers can build EnsembleCalculator without a GPU).
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model_paths: list[str], device: str = "cuda") -> None:
        super().__init__()
        self.model_paths = list(model_paths)
        self.device = device
        self._members: Optional[list[Calculator]] = None

    def _load_members(self) -> list[Calculator]:
        if self._members is None:
            from nequip.ase import NequIPCalculator
            self._members = [
                NequIPCalculator.from_compiled_model(str(p), device=self.device)
                for p in self.model_paths
            ]
        return self._members

    def calculate(self, atoms: Optional[Atoms] = None,
                  properties: Optional[list[str]] = None,
                  system_changes: Optional[list[str]] = None) -> None:
        if system_changes is None:
            system_changes = all_changes
        super().calculate(atoms, properties, system_changes)
        members = self._load_members()
        Es, Fs, Ss = [], [], []
        for m in members:
            ac = atoms.copy()
            ac.calc = m
            Es.append(float(ac.get_potential_energy()))
            Fs.append(np.array(ac.get_forces()))
            try:
                Ss.append(np.array(ac.get_stress()))
            except Exception:
                Ss.append(np.full(6, np.nan))
        self._last_E = np.array(Es)
        self._last_F = np.stack(Fs, axis=0)         # (K, n_atoms, 3)
        self._last_S = np.stack(Ss, axis=0)         # (K, 6)

        self.results["energy"] = float(self._last_E.mean())
        self.results["forces"] = self._last_F.mean(axis=0)
        if not np.any(np.isnan(self._last_S)):
            self.results["stress"] = self._last_S.mean(axis=0)

    def get_uncertainty(self, atoms: Atoms) -> dict[str, float]:
        """Return per-structure σ_E, σ_F (mean and max over atoms), σ_S
        (mean over Voigt components)."""
        # Trigger calculation if needed
        if (self.atoms is None or
                self.atoms.positions.shape != atoms.positions.shape or
                not np.allclose(self.atoms.positions, atoms.positions)):
            self.calculate(atoms, properties=["energy", "forces", "stress"])
        sigma_E = float(self._last_E.std())
        F_mean = self._last_F.mean(axis=0)
        F_dev = self._last_F - F_mean[None]
        # σ_F per atom = std-dev of (F_k - F_mean) over members
        atom_sigma = np.linalg.norm(F_dev, axis=2).std(axis=0)  # (n_atoms,)
        sigma_F_mean = float(atom_sigma.mean())
        sigma_F_max = float(atom_sigma.max())
        sigma_S = float(self._last_S.std(axis=0).mean())
        return {
            "sigma_E": sigma_E,
            "sigma_F_mean": sigma_F_mean,
            "sigma_F_max": sigma_F_max,
            "sigma_S": sigma_S,
        }
