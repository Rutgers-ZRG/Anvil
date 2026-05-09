"""Pool C — Anchor: foundation-MLIP-only regime-aligned structures.

See DESIGN.md §3.4.

In the v1 piece-3 minimum, this module is foundation-MLIP-FREE: it generates
strain probes by applying isotropic factors to the seed structures directly,
without prior MLIP relaxation. Hot MD probes and prior relaxation come in
week 2 once the foundation MLIP loader is wired.

For each (phase × P) pair, applies isotropic strain factors (default
{0.95, 0.97, 1.03, 1.05}) → 4 strain cells per (phase × P).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ase import Atoms
from ase.io import read as ase_read


@dataclass
class AnchorConfig:
    phases: dict[str, str | Path]
    pressures_gpa: list[float]
    temperatures_k: list[float] = field(default_factory=list)
    strain_factors: list[float] = field(
        default_factory=lambda: [0.95, 0.97, 1.03, 1.05]
    )
    md_steps: int = 100
    md_snapshots: int = 5


def _apply_isotropic_strain(atoms: Atoms, factor: float) -> Atoms:
    """Multiply all lattice vectors by `factor`; scale fractional coords."""
    a = atoms.copy()
    new_cell = a.get_cell() * factor
    a.set_cell(new_cell, scale_atoms=True)
    return a


def _read_seed(path: str | Path) -> Atoms:
    """Read a POSCAR (or any ASE-recognized structure file)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Seed not found: {p}")
    return ase_read(str(p))


def generate_pool_c(
    config: AnchorConfig,
    *,
    n_target: int | None = None,
) -> list[Atoms]:
    """Generate Pool C strain-probe candidates.

    Per (phase, P) pair: read seed POSCAR, apply each strain factor, tag
    metadata. Returns list of ASE Atoms with info fields:
        atoms.info = {
            "pool": "C",
            "phase": <name>,
            "pressure_gpa": <float>,
            "sample_kind": f"strain_{factor:.2f}",
        }

    NOTE: v1 piece-3 minimum. Does not relax under target pressure (the
    "pressure" tag identifies the bin in the regime grid, not the actual
    relaxed pressure). DFT labeling will compute the true stress; that's
    what the model trains on.

    Future (week 2): foundation-MLIP relax at each P first, then strain.
    """
    out: list[Atoms] = []
    for phase, seed_path in config.phases.items():
        seed = _read_seed(seed_path)
        for P in config.pressures_gpa:
            for factor in config.strain_factors:
                a = _apply_isotropic_strain(seed, factor)
                a.info["pool"] = "C"
                a.info["phase"] = phase
                a.info["pressure_gpa"] = float(P)
                a.info["sample_kind"] = f"strain_{factor:.2f}"
                out.append(a)
    if n_target is not None and len(out) > n_target:
        # Subsample uniformly
        import numpy as np
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(out), n_target, replace=False))
        out = [out[i] for i in idx]
    return out
