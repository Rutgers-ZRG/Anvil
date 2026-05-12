"""Pool A — Per-regime stratified entropy MD orchestrator.

See DESIGN.md §3.2.

For each (phase × P × T × k_factor × mode):
  1. Foundation-MLIP relax to (P, T)  (FIRE + cell-filter under hydrostatic P)
  2. Thermalize at T (Maxwell-Boltzmann + brief equilibration)
  3. Langevin NVT MD wrapped by GlobalEntropyCalculator
  4. Snapshots every md_steps/n_snapshots; accept iff entropy_gain > min_gain
  5. Accepted snapshots are added to the shared FingerprintDataset so the
     remainder of MD sees them as already-known.

Each cell shares one FingerprintDataset across phases / pressures /
temperatures / k_factors / modes — global novelty across the full grid.
Starting MD from each (phase, P, T) point ensures every regime gets
entropy-novel candidates representative of that regime.

Local entropy chaining (reformpy.EntropyMaximizingCalculator) is supported
optionally; defaults to global-only because Subramanyam/Perez 2025 showed
global entropy dominates the contribution.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read as ase_read

from anvil.al.generation.global_entropy import (
    FingerprintDataset,
    GlobalEntropyCalculator,
)


@dataclass
class StratifiedMDConfig:
    """Configuration for the per-regime entropy MD grid."""
    phases: dict[str, str]                   # phase_name -> POSCAR path
    pressures_gpa: list[float]
    temperatures_k: list[float]
    k_factors: list[float] = field(default_factory=lambda: [0, 2, 5])
    modes: list[str] = field(default_factory=lambda: ["per_atom", "per_config"])
    md_steps: int = 1000
    md_dt_fs: float = 1.0
    md_friction: float = 0.01                # Langevin friction (1/fs)
    n_snapshots: int = 10
    fp_cutoff: float = 5.0
    fp_natx: int = 50
    regularization: float = 1e-3
    min_entropy_gain: float = 0.01
    relax_fmax: float = 0.05                 # eV/Å for foundation relax
    relax_max_steps: int = 200
    use_local_entropy: bool = False
    local_k_factor: float = 1.0
    local_cutoff: float = 4.0


def _relax_at_pressure(
    atoms: Atoms,
    pressure_gpa: float,
    calc: Calculator,
    *,
    fmax: float = 0.05,
    steps: int = 200,
) -> Atoms:
    """Foundation-MLIP relax under hydrostatic pressure.

    Uses FrechetCellFilter when available (ASE ≥ 3.23), falls back to
    ExpCellFilter on older versions.
    """
    from ase.optimize import FIRE
    try:
        from ase.filters import FrechetCellFilter as CellFilter
    except ImportError:
        from ase.filters import ExpCellFilter as CellFilter

    P_eV_per_A3 = pressure_gpa / 160.21766208
    a = atoms.copy()
    a.calc = calc
    sf = CellFilter(a, scalar_pressure=P_eV_per_A3)
    opt = FIRE(sf, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return a


def _thermalize_and_md_with_callback(
    atoms: Atoms,
    T_K: float,
    calc: Calculator,
    *,
    n_steps: int,
    dt_fs: float,
    friction: float,
    snapshot_every: int,
    on_snapshot,
) -> None:
    """Run Langevin MD from a relaxed cell, calling `on_snapshot(atoms)` every
    `snapshot_every` steps.

    `on_snapshot` is responsible for any deepcopy + accept/reject logic.
    """
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.units import fs as fs_unit

    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_K)
    dyn = Langevin(atoms, dt_fs * fs_unit, temperature_K=T_K, friction=friction)
    dyn.attach(lambda: on_snapshot(atoms), interval=snapshot_every)
    dyn.run(n_steps)


def generate_pool_a(
    config: StratifiedMDConfig,
    base_calc: Calculator,
    *,
    n_target: Optional[int] = None,
    fp_dataset_path: Optional[Path] = None,
    seed_atoms_per_phase: Optional[dict[str, Atoms]] = None,
    verbose: bool = True,
) -> tuple[list[Atoms], FingerprintDataset]:
    """Generate Pool A candidates by per-regime stratified entropy MD.

    Args:
        config: Stratified MD grid + entropy parameters.
        base_calc: Foundation MLIP ASE calculator (e.g. NequIPCalculator).
        n_target: Hard cap on output candidates (subsample if exceeded).
        fp_dataset_path: If provided, load existing FingerprintDataset state
            from this path at the start (closed-loop awareness across rounds);
            save the updated state at the end.
        seed_atoms_per_phase: Override seed Atoms objects per phase (otherwise
            read from config.phases[phase] POSCAR paths).

    Returns:
        (candidates, fp_dataset). The candidates carry metadata:
            atoms.info = {pool: "A", phase, pressure_gpa, temperature_k,
                          k_factor, mode, sample_kind: "entropy_md_..."}
    """
    # Load or initialize FingerprintDataset
    if fp_dataset_path is not None and Path(fp_dataset_path).exists():
        ds = FingerprintDataset.load(str(fp_dataset_path))
        if ds.fp_dim != config.fp_natx:
            raise ValueError(
                f"FingerprintDataset fp_dim ({ds.fp_dim}) ≠ config.fp_natx "
                f"({config.fp_natx})"
            )
    else:
        ds = FingerprintDataset(fp_dim=config.fp_natx, reg=config.regularization)

    # Resolve seed Atoms per phase
    seeds: dict[str, Atoms] = {}
    if seed_atoms_per_phase:
        seeds.update(seed_atoms_per_phase)
    for name, path in config.phases.items():
        if name not in seeds:
            seeds[name] = ase_read(str(path))

    save_every = max(1, config.md_steps // config.n_snapshots)
    candidates: list[Atoms] = []

    for phase, seed in seeds.items():
        for P in config.pressures_gpa:
            if verbose:
                print(f"[Pool A] Relaxing {phase} at {P} GPa...")
            try:
                relaxed = _relax_at_pressure(
                    seed, P, base_calc,
                    fmax=config.relax_fmax, steps=config.relax_max_steps,
                )
            except Exception as exc:
                if verbose:
                    print(f"[Pool A]   relax FAILED for ({phase}, {P} GPa): {exc}")
                continue

            for T in config.temperatures_k:
                for k_factor in config.k_factors:
                    for mode in config.modes:
                        if verbose:
                            print(f"[Pool A]   {phase} P={P} T={T}K k={k_factor} "
                                  f"mode={mode}")
                        # Build the entropy-wrapped calculator
                        entropy_calc = GlobalEntropyCalculator(
                            calculator=base_calc,
                            dataset=ds,
                            k_factor=k_factor,
                            cutoff=config.fp_cutoff,
                            natx=config.fp_natx,
                            mode=mode,
                        )
                        # Optional local entropy chain
                        if config.use_local_entropy:
                            try:
                                from reformpy import EntropyMaximizingCalculator
                            except ImportError as exc:
                                raise RuntimeError(
                                    "config.use_local_entropy requires reformpy"
                                ) from exc
                            entropy_calc = EntropyMaximizingCalculator(
                                calculator=entropy_calc,
                                k_factor=config.local_k_factor,
                                cutoff=config.local_cutoff,
                                natx=None,
                            )

                        atoms = relaxed.copy()
                        cell_candidates: list[Atoms] = []

                        # If min_entropy_gain is set very negative (e.g. -1e9),
                        # bypass entropy_gain computation entirely — accept all
                        # snapshots, rely on n_target subsampling. Cast to float
                        # because yaml can parse "-1.0e9" as str.
                        try:
                            _meg = float(config.min_entropy_gain)
                        except (TypeError, ValueError):
                            _meg = 0.0
                        bypass_filter = _meg <= -1e6

                        def on_snapshot(at: Atoms,
                                        _p=phase, _P=P, _T=T,
                                        _k=k_factor, _m=mode) -> None:
                            if bypass_filter:
                                gain = 0.0
                            else:
                                try:
                                    gain = entropy_calc.entropy_gain(at)
                                except Exception as exc:
                                    if verbose:
                                        print(f"[Pool A]   entropy_gain "
                                              f"raised {type(exc).__name__}: "
                                              f"{exc}; dropping snapshot")
                                    return
                            if bypass_filter or gain > _meg:
                                snap = at.copy()
                                snap.calc = None
                                snap.info["pool"] = "A"
                                snap.info["phase"] = _p
                                snap.info["pressure_gpa"] = float(_P)
                                snap.info["temperature_k"] = float(_T)
                                snap.info["k_factor"] = float(_k)
                                snap.info["entropy_mode"] = _m
                                snap.info["entropy_gain"] = float(gain)
                                snap.info["sample_kind"] = (
                                    f"entropy_md_k{_k:g}_{_m}"
                                )
                                cell_candidates.append(snap)
                                if not bypass_filter:
                                    # Add to shared dataset so future MD steps see it
                                    try:
                                        entropy_calc.add_config(at)
                                    except Exception:
                                        pass

                        try:
                            _thermalize_and_md_with_callback(
                                atoms, T, entropy_calc,
                                n_steps=config.md_steps,
                                dt_fs=config.md_dt_fs,
                                friction=config.md_friction,
                                snapshot_every=save_every,
                                on_snapshot=on_snapshot,
                            )
                        except Exception as exc:
                            if verbose:
                                print(f"[Pool A]     MD failed: {exc}")

                        candidates.extend(cell_candidates)
                        if verbose:
                            print(f"[Pool A]     accepted {len(cell_candidates)} "
                                  f"snapshots (running total {len(candidates)})")

    # Subsample to n_target uniformly across (phase, P, T, k, mode) bins
    if n_target is not None and len(candidates) > n_target:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(candidates), n_target, replace=False))
        candidates = [candidates[i] for i in idx]

    # Persist FP-dataset state for next round
    if fp_dataset_path is not None:
        Path(fp_dataset_path).parent.mkdir(parents=True, exist_ok=True)
        ds.save(str(fp_dataset_path))

    return candidates, ds
