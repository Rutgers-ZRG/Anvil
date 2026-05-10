"""Unit tests for Pool B reform-relax with a stub calculator.

The full reform-relax workflow needs reformpy.Reform_Calculator (which imports
mpi4py); we limit unit tests to the perturbation helper + symmetry filter +
SumCalculator. The full pipeline is exercised on amareln.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write as ase_write

from anvil.al.generation.reform_relax import (
    ReformRelaxConfig,
    _SumCalculator,
    _has_finite_symmetry,
    _perturb,
)


class StubCalc(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, e=-1.0, **kw):
        super().__init__(**kw)
        self.e = e

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        self.results = {
            "energy": self.e * n,
            "forces": np.zeros((n, 3)),
            "stress": np.zeros(6),
        }


def test_perturb_changes_positions_and_cell() -> None:
    atoms = Atoms("Si", positions=[(0, 0, 0)],
                  cell=[(3, 0, 0), (0, 3, 0), (0, 0, 3)], pbc=True)
    rng = np.random.default_rng(0)
    p = _perturb(atoms, rattle=0.5, cell=0.05, rng=rng)
    assert not np.allclose(p.get_positions(), atoms.get_positions())
    assert not np.allclose(p.get_cell()[:], atoms.get_cell()[:])


def test_sum_calculator_energy() -> None:
    atoms = Atoms("Si2",
                  positions=[(0, 0, 0), (1, 1, 1)],
                  cell=[(4, 0, 0), (0, 4, 0), (0, 0, 4)], pbc=True)
    c1 = StubCalc(e=-1.0)
    c2 = StubCalc(e=-3.0)
    combined = _SumCalculator([c1, c2], [1.0, 0.5])
    atoms.calc = combined
    e = atoms.get_potential_energy()
    assert e == pytest.approx(-1.0 * 2 + 0.5 * (-3.0 * 2))   # = -2 + -3 = -5
    f = atoms.get_forces()
    assert np.allclose(f, 0.0)
    s = atoms.get_stress()
    assert np.allclose(s, 0.0)


def test_has_finite_symmetry_diamond() -> None:
    """Si in cubic-diamond is high symmetry (Fd-3m, # 227)."""
    a = 5.43
    atoms = Atoms(
        symbols=["Si"] * 8,
        scaled_positions=[
            (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
            (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
            (0.75, 0.25, 0.75), (0.25, 0.75, 0.75),
        ],
        cell=[(a, 0, 0), (0, a, 0), (0, 0, a)], pbc=True,
    )
    pytest.importorskip("spglib")
    assert _has_finite_symmetry(atoms)


def test_has_finite_symmetry_p1_returns_false() -> None:
    """Random distorted cell is P1 → drop."""
    pytest.importorskip("spglib")
    rng = np.random.default_rng(0)
    a = 5.0
    n = 8
    atoms = Atoms(
        symbols=["Si"] * n,
        scaled_positions=rng.uniform(0, 1, size=(n, 3)),
        cell=[(a, 0.1, 0.2), (0.05, a, 0.0), (0.07, 0.03, a)], pbc=True,
    )
    assert not _has_finite_symmetry(atoms, symprec=0.001)


def test_config_defaults() -> None:
    cfg = ReformRelaxConfig()
    assert cfg.mode == "stability"
    assert cfg.perturbations_per_seed == 5
    assert cfg.reform_lambda == 1.0
    cfg2 = ReformRelaxConfig(mode="discovery")
    assert cfg2.mode == "discovery"
