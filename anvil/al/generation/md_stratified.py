"""Pool A — Per-regime stratified entropy MD orchestrator.

See DESIGN.md §3.2.

For each (phase × P × T × k_factor × mode):
  - Foundation-MLIP relax to (P, T)
  - Thermalize at T
  - MD steps with EntropyMaximizingCalculator (local) AND
    GlobalEntropyCalculator (global) chained
  - Accept candidates via FingerprintDataset.entropy_gain > min_gain

The per-condition stratification logic ports relevant slices from
/Users/li/dev/RA/mlip-active-learn/generate_structures.py.

Each cell shares one FingerprintDataset across phases / pressures /
temperatures to maintain global novelty, but starts each MD trajectory from
the relaxed phase at its target (P, T) — this guarantees every regime
produces entropy-novel candidates representative of that regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator


@dataclass
class StratifiedMDConfig:
    phases: dict[str, str]                  # phase_name -> POSCAR path
    pressures_gpa: list[float]
    temperatures_k: list[float]
    k_factors: list[float] = field(default_factory=lambda: [0, 2, 5])
    modes: list[str] = field(default_factory=lambda: ["per_atom", "per_config"])
    md_steps: int = 1000
    fp_cutoff: float = 5.0
    fp_natx: int = 50
    regularization: float = 1e-3
    min_entropy_gain: float = 0.01


def generate_pool_a(
    config: StratifiedMDConfig,
    base_calc: Calculator,
    *,
    n_target: int,
    fp_dataset_path: Path | None = None,
) -> list[Atoms]:
    """Run the (phase × P × T × k × mode) grid; collect candidates passing the
    entropy-gain accept threshold; return up to n_target Atoms.

    fp_dataset_path: if provided, load and persist the FingerprintDataset
    state across rounds (closed-loop dataset awareness).
    """
    raise NotImplementedError("week 1: ports generate_structures.py logic")
