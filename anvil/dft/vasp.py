"""VASP input generation. INCAR templates per functional, KPOINTS by k-density.

See DESIGN.md §4.3.

Templates inherit from PI's *_setup_vasp.py scripts in mlip-active-learn:
- carbon_setup_vasp.py (r2SCAN+rVV10)
- si_setup_vasp.py    (PBE)
- nacl_setup_vasp.py  (PBE with explicit Na sv POTCAR)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase import Atoms


@dataclass
class VASPInputs:
    """Bundle of VASP input files for one structure."""
    incar: str
    kpoints: str
    poscar: str
    potcar_paths: list[str]


def make_vasp_inputs(
    atoms: Atoms,
    functional: str,
    encut: int = 600,
    kspacing: float = 0.25,
    extra_incar: dict | None = None,
    max_kpts: int = 500,
) -> VASPInputs:
    """Build INCAR, KPOINTS, POSCAR, and POTCAR list for the structure.

    `functional` selects the INCAR template under configs/functionals/.
    Implements PI's conventions:
      - Do NOT use LREAL=Auto.
      - For r2SCAN+rVV10: METAGGA=R2SCAN, BPARAM=11.95, LASPH=.TRUE., NPAR=4.
      - K-density with max_total_kpts cap of 500.
    """
    raise NotImplementedError("week 1: see configs/functionals/*.yaml")


def write_vasp_dir(
    atoms: Atoms,
    out_dir: str | Path,
    inputs: VASPInputs,
    *,
    sample_kind: str | None = None,
    pool: str | None = None,
) -> Path:
    """Write INCAR/KPOINTS/POSCAR/POTCAR + Slurm sbp.sh to out_dir.

    Tags `pool` ∈ {A, B, C, validation} into a small JSON metadata file alongside
    so the collector can route the result correctly.
    """
    raise NotImplementedError("week 1")
