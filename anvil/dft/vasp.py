"""VASP input generation. INCAR templates per functional, KPOINTS by k-density.

See DESIGN.md §4.3.

Templates inherit from PI's *_setup_vasp.py scripts in mlip-active-learn:
- carbon_setup_vasp.py (r2SCAN+rVV10)
- si_setup_vasp.py    (PBE)
- nacl_setup_vasp.py  (PBE with explicit Na sv POTCAR)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from ase import Atoms
from ase.io import write as ase_write


# Per-element POTCAR resolution. Default = element symbol (e.g. "C/POTCAR");
# override here for elements that need a non-trivial POTCAR (e.g. Na_sv).
POTCAR_OVERRIDES: dict[str, str] = {
    "Na": "Na_sv",
    "K": "K_sv",
    "Rb": "Rb_sv",
    "Cs": "Cs_sv",
}


def _functional_yaml_path(functional: str) -> Path:
    """Resolve configs/functionals/<functional>.yaml from package data."""
    here = Path(__file__).resolve().parent
    candidate = here.parent.parent / "configs" / "functionals" / f"{functional}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"No INCAR template for functional {functional!r} at {candidate}. "
            f"Add a yaml to configs/functionals/."
        )
    return candidate


def load_functional(functional: str) -> dict:
    """Read configs/functionals/<functional>.yaml and return the parsed dict."""
    with open(_functional_yaml_path(functional)) as f:
        return yaml.safe_load(f)


@dataclass
class VASPInputs:
    """Bundle of VASP input file CONTENTS (strings) for one structure."""
    incar: str
    kpoints: str
    poscar: str
    potcar_paths: list[str]
    sample_kind: str = ""                  # e.g. "strain_0.95"
    pool: str = ""                          # "C" | "validation" | etc.


def render_incar(template: dict, encut: int, extra: dict | None = None) -> str:
    """Render INCAR text from a yaml-defined incar dict.

    Required values that user can override (encut, etc.) come from args;
    template-only values pass through verbatim. Template values are dumped
    in input order; vendor convention places SYSTEM first.
    """
    merged: dict = dict(template)
    merged["ENCUT"] = encut
    if extra:
        merged.update(extra)
    lines = []
    for k, v in merged.items():
        if isinstance(v, bool):
            v = ".TRUE." if v else ".FALSE."
        lines.append(f"{k} = {v}")
    return "\n".join(lines) + "\n"


def render_kpoints(
    atoms: Atoms,
    kspacing: float = 0.25,
    *,
    max_total_kpts: int = 500,
) -> str:
    """Generate gamma-centered KPOINTS by reciprocal-space density.

    `kspacing` is the target spacing in 1/Å (smaller → denser mesh).
    `max_total_kpts` caps the total count for degenerate cells (PI convention).
    """
    cell = atoms.get_cell()
    rec = np.linalg.norm(cell.reciprocal(), axis=1)
    # Number of points along each axis: 2π / (kspacing × |b_i|) ≈ 1/(kspacing |a_i| / 2π)
    # PI's convention in mlip-active-learn uses density = 40 (rough KPPRA-like).
    # Map kspacing → density via density = 2π / kspacing
    density = max(2.0 * np.pi / kspacing, 1.0)
    kpts = [max(1, int(np.ceil(density * r))) for r in rec]
    total = kpts[0] * kpts[1] * kpts[2]
    if total > max_total_kpts:
        scale = (max_total_kpts / total) ** (1.0 / 3.0)
        kpts = [max(1, int(round(k * scale))) for k in kpts]
    return (
        "Automatic mesh\n"
        "0\n"
        "Gamma\n"
        f"  {kpts[0]}  {kpts[1]}  {kpts[2]}\n"
        "  0  0  0\n"
    )


def render_poscar(atoms: Atoms) -> str:
    """ASE POSCAR (vasp5, fractional coords)."""
    from io import StringIO
    buf = StringIO()
    ase_write(buf, atoms, format="vasp", vasp5=True, direct=True)
    return buf.getvalue()


def potcar_paths_for(elements: Iterable[str], potcar_root: str | Path) -> list[str]:
    """Return concatenation-ordered list of POTCAR paths for elements.

    Uses POTCAR_OVERRIDES for elements like Na → Na_sv.
    """
    paths = []
    for el in elements:
        sub = POTCAR_OVERRIDES.get(el, el)
        p = Path(potcar_root) / sub / "POTCAR"
        paths.append(str(p))
    return paths


def make_vasp_inputs(
    atoms: Atoms,
    functional: str,
    *,
    encut: int = 600,
    kspacing: float = 0.25,
    extra_incar: dict | None = None,
    potcar_root: str | Path = "/home/lz432/apps/PBE64",
    sample_kind: str = "",
    pool: str = "",
) -> VASPInputs:
    """Build INCAR / KPOINTS / POSCAR / POTCAR list for one structure."""
    template = load_functional(functional)
    incar = render_incar(template["incar"], encut=encut, extra=extra_incar)
    kpoints = render_kpoints(atoms, kspacing=kspacing)
    poscar = render_poscar(atoms)

    # Element order from the POSCAR — ASE writes by group of unique elements
    syms = list(dict.fromkeys(atoms.get_chemical_symbols()))
    pots = potcar_paths_for(syms, potcar_root)

    return VASPInputs(
        incar=incar, kpoints=kpoints, poscar=poscar, potcar_paths=pots,
        sample_kind=sample_kind, pool=pool,
    )


def write_vasp_dir(
    inputs: VASPInputs,
    out_dir: str | Path,
    *,
    cat_potcar: bool = True,
) -> Path:
    """Write INCAR/KPOINTS/POSCAR + concatenated POTCAR + .anvil_meta.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "INCAR").write_text(inputs.incar)
    (out / "KPOINTS").write_text(inputs.kpoints)
    (out / "POSCAR").write_text(inputs.poscar)
    if cat_potcar:
        with open(out / "POTCAR", "w") as f:
            for p in inputs.potcar_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"POTCAR missing at {p}")
                with open(p) as src:
                    f.write(src.read())
    meta = {
        "sample_kind": inputs.sample_kind,
        "pool": inputs.pool,
        "potcar_paths": inputs.potcar_paths,
    }
    (out / ".anvil_meta.json").write_text(json.dumps(meta, indent=2))
    return out
