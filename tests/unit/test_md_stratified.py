"""Unit tests for Pool A entropy MD orchestrator with a stub calculator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write as ase_write

# Skip if libfp / fingerprint stack unavailable
pytest.importorskip("libfp")

from anvil.al.generation.md_stratified import (
    StratifiedMDConfig,
    _relax_at_pressure,
    generate_pool_a,
)


class StubCalc(Calculator):
    """Cheap deterministic ASE calculator for unit tests.

    Returns zero forces, zero stress, constant energy. Pool A's entropy term
    will dominate any "force" the model sees, so the MD just bounces around
    the entropy gradient.
    """
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        self.results = {
            "energy": -1.0 * n,
            "forces": np.zeros((n, 3)),
            "stress": np.zeros(6),
        }


def _make_seed(tmp_path: Path) -> Path:
    a = 5.0
    atoms = Atoms(
        symbols=["Si"] * 8,
        positions=[
            (0, 0, 0), (a/2, a/2, 0), (a/2, 0, a/2), (0, a/2, a/2),
            (a/4, a/4, a/4), (3*a/4, 3*a/4, a/4),
            (3*a/4, a/4, 3*a/4), (a/4, 3*a/4, 3*a/4),
        ],
        cell=[(a, 0, 0), (0, a, 0), (0, 0, a)],
        pbc=True,
    )
    p = tmp_path / "si.vasp"
    ase_write(str(p), atoms, format="vasp", vasp5=True, direct=True)
    return p


def test_relax_at_pressure_runs(tmp_path: Path) -> None:
    """Smoke: relax_at_pressure does not throw on a stub calc + Si cell."""
    p = _make_seed(tmp_path)
    from ase.io import read
    atoms = read(str(p))
    relaxed = _relax_at_pressure(atoms, 0.0, StubCalc(), fmax=0.05, steps=5)
    assert len(relaxed) == 8
    assert relaxed.get_volume() > 0


def test_pool_a_zero_k_short_circuit(tmp_path: Path) -> None:
    """With k_factor=0, GlobalEntropyCalculator short-circuits and Pool A MD
    behaves like vanilla foundation MD — but on a stub calc the FP cost still
    dominates. Use this test to verify the API surface plus k=0 path; the
    real integration smoke is on amareln with the foundation MLIP.
    """
    from anvil.al.generation.global_entropy import GlobalEntropyCalculator
    from anvil.al.generation.global_entropy import FingerprintDataset
    ds = FingerprintDataset(fp_dim=16, reg=1e-3)
    calc = GlobalEntropyCalculator(
        calculator=StubCalc(), dataset=ds,
        k_factor=0.0, cutoff=4.0, natx=16, mode="per_atom",
    )
    # k=0 short-circuit: no FP compute, immediate base-calc passthrough
    seed_path = _make_seed(tmp_path)
    from ase.io import read
    atoms = read(str(seed_path))
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert e == -8.0       # 8-atom Si × stub -1.0/atom


# NOTE: The full Pool A grid (foundation MLIP relax + Langevin entropy MD)
# is exercised as an INTEGRATION test on amareln (see scripts/smoke_pool_a.py
# in week-2 deliverables). Unit-testing it with a stub calc is impractical
# because libfp dominates wallclock and the stub adds no MD signal.
