"""Pool B — Reform-relax (CRISP-style, FP-equivalence regularized).

See DESIGN.md §3.3.

Reformpy's Reform_Calculator implements an FP-variance regularizer:

    E_reform = sum_(i,j of same type) ||fp_i - fp_j||^2

When chained with a foundation MLIP via SumCalculator with weight λ,
relaxation drives toward structures where same-type atoms have similar
fingerprints — i.e., crystalline-ordered phases. This is the CRISP
mechanism applied as an active-learning pool.

Mode (a) "stability" — auto_clusters=False:
    Single per-type cluster → drives toward fully ordered single-environment
    crystals (e.g. fcc, diamond). Default.

Mode (b) "discovery" — auto_clusters=True with k-means clustering on FPs:
    Allows multiple distinct local environments per type → drives toward
    multi-environment crystals like γ-B28 or perovskites. Opt-in via
    config.mode = "discovery".

In v1, initial structures are perturbed seed phases only. pyXtal random
generation + Pool A low-F snapshot integration are deferred to week 3.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read as ase_read
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class ReformRelaxConfig:
    """Configuration for Pool B reform-relax generation."""
    mode: Literal["stability", "discovery"] = "stability"
    perturbations_per_seed: int = 5
    perturb_rattle: float = 0.3                   # Å, atomic rattle stddev
    perturb_cell: float = 0.05                   # fractional, isotropic cell perturb
    relax_max_steps: int = 200
    relax_fmax: float = 0.05                     # eV/Å
    sample_intermediates: int = 2
    spglib_filter: bool = True
    pyxtal_n_per_round: int = 0                  # v3 (week 3)
    pallas_pathway_xyz: Optional[str] = None     # v2 hook

    # Reform-specific
    reform_lambda: float = 1.0                   # weight on FP-variance term
    reform_cutoff: float = 4.0
    reform_nx: int = 300
    reform_lmax: int = 0
    discovery_min_k: int = 2
    discovery_max_k: int = 8


class _SumCalculator(Calculator):
    """Lightweight sum-of-calculators (avoids reformpy.mixing's mpi4py import).

    E = Σ w_i E_i
    F = Σ w_i F_i
    σ = Σ w_i σ_i
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, calcs, weights):
        super().__init__()
        self.calcs = list(calcs)
        self.weights = [float(w) for w in weights]
        if len(self.calcs) != len(self.weights):
            raise ValueError("calcs and weights length mismatch")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e_total = 0.0
        f_total = None
        s_total = None
        for calc, w in zip(self.calcs, self.weights):
            ac = atoms.copy()
            ac.calc = calc
            e_total += w * float(ac.get_potential_energy())
            f = np.array(ac.get_forces())
            f_total = w * f if f_total is None else f_total + w * f
            try:
                s = np.array(ac.get_stress())
                s_total = w * s if s_total is None else s_total + w * s
            except (PropertyNotImplementedError, NotImplementedError):
                pass
        self.results["energy"] = e_total
        self.results["forces"] = f_total
        if s_total is not None:
            self.results["stress"] = s_total


def _perturb(atoms: Atoms, rattle: float, cell: float, *, rng: np.random.Generator) -> Atoms:
    """Random rattle on positions + isotropic cell perturbation."""
    a = atoms.copy()
    pos = a.get_positions()
    pos = pos + rng.normal(scale=rattle, size=pos.shape)
    a.set_positions(pos)
    factor = 1.0 + rng.uniform(-cell, cell)
    new_cell = a.get_cell() * factor
    a.set_cell(new_cell, scale_atoms=True)
    return a


def _has_finite_symmetry(atoms: Atoms, *, symprec: float = 0.05) -> bool:
    """True if spglib detects symmetry above P1.

    Returns True (keep) if spglib unavailable — we never silently drop on
    missing optional dependencies.
    """
    try:
        import spglib
    except ImportError:
        return True
    cell = (atoms.get_cell()[:], atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    try:
        ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    except Exception:
        return True
    if ds is None:
        return False
    return ds["number"] > 1


def _build_reform_calc(atoms: Atoms, cfg: ReformRelaxConfig) -> Calculator:
    """Construct the Reform_Calculator for the given atoms object.

    Passes an explicit SerialComm so reformpy doesn't try to load MPI at
    Reform_Calculator init time — important on GPU nodes whose runtime
    environment doesn't expose libmpi.
    """
    from reformpy import Reform_Calculator
    from reformpy.calculator import SerialComm

    znums = sorted(set(atoms.get_atomic_numbers()))
    ntyp = len(znums)
    auto_clusters = cfg.mode == "discovery"
    return Reform_Calculator(
        atoms=atoms,
        cutoff=cfg.reform_cutoff,
        nx=cfg.reform_nx,
        lmax=cfg.reform_lmax,
        ntyp=ntyp,
        znucl=np.array(znums, dtype=np.int32),
        auto_clusters=auto_clusters,
        min_k=cfg.discovery_min_k,
        max_k=cfg.discovery_max_k,
        parallel=False,
        comm=SerialComm(),
    )


def generate_pool_b(
    config: ReformRelaxConfig,
    base_calc: Calculator,
    seed_phases: dict[str, str],
    *,
    n_target: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> list[Atoms]:
    """Generate Pool B candidates.

    Workflow:
      1. For each seed phase, generate `perturbations_per_seed` rattled+strained
         initial structures.
      2. Wrap base_calc + Reform_Calculator via _SumCalculator.
      3. FIRE relax until fmax < relax_fmax or relax_max_steps.
      4. Optional spglib filter (drop P1 cells).
      5. Sample final relaxed cell as candidate.

    v1 scope: only perturbed seeds. pyXtal random + Pool A low-F integration
    deferred to week 3.
    """
    from ase.optimize import FIRE

    rng = np.random.default_rng(seed)
    candidates: list[Atoms] = []

    for phase, path in seed_phases.items():
        seed_atoms = ase_read(str(path))
        for i in range(config.perturbations_per_seed):
            init = _perturb(seed_atoms, config.perturb_rattle, config.perturb_cell, rng=rng)
            try:
                reform = _build_reform_calc(init, config)
            except Exception as exc:
                if verbose:
                    print(f"[Pool B] reform-build failed ({phase}, perturb {i}): {exc}")
                continue
            combined = _SumCalculator([base_calc, reform], [1.0, config.reform_lambda])

            atoms = init.copy()
            atoms.calc = combined
            try:
                opt = FIRE(atoms, logfile=None)
                opt.run(fmax=config.relax_fmax, steps=config.relax_max_steps)
            except Exception as exc:
                if verbose:
                    print(f"[Pool B]   relax failed ({phase}, perturb {i}): {exc}")
                continue

            if config.spglib_filter and not _has_finite_symmetry(atoms):
                if verbose:
                    print(f"[Pool B]   {phase} perturb {i}: P1 — dropped")
                continue

            final = atoms.copy()
            final.calc = None
            final.info["pool"] = "B"
            final.info["phase"] = phase
            final.info["sample_kind"] = f"reform_relax_{config.mode}"
            final.info["reform_lambda"] = float(config.reform_lambda)
            final.info["reform_mode"] = config.mode
            candidates.append(final)
            if verbose:
                print(f"[Pool B]   {phase} perturb {i}: relaxed → kept "
                      f"({len(atoms)} atoms)")

    # v1 has no pyXtal / Pool A inputs yet.
    # FP-distance dedupe also deferred to week 3.

    if n_target is not None and len(candidates) > n_target:
        rng = np.random.default_rng(seed + 1)
        idx = sorted(rng.choice(len(candidates), n_target, replace=False))
        candidates = [candidates[i] for i in idx]

    return candidates
