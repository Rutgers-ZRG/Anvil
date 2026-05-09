"""Pool C — Anchor: foundation-MLIP-only regime-aligned structures.

See DESIGN.md §3.4.

Per (phase × P): 1 equilibrium + 4 strain probes + |T| × 5 thermal MD snaps.
The strain factors {0.95, 0.97, 1.03, 1.05} are SMALLER than the indtest
{0.92, 1.08} — anchor is training data, not validation. Stays close to
equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator


@dataclass
class AnchorConfig:
    phases: dict[str, str]
    pressures_gpa: list[float]
    temperatures_k: list[float]
    strain_factors: list[float] = field(
        default_factory=lambda: [0.95, 0.97, 1.03, 1.05]
    )
    md_steps: int = 100              # short — populates near-eq regime only
    md_snapshots: int = 5


def generate_pool_c(
    config: AnchorConfig,
    foundation_calc: Calculator,
    *,
    n_target: int,
) -> list[Atoms]:
    """For each (phase, P): foundation_relax + strain × 4 + thermal-MD × |T|.

    Returns up to n_target structures from the regime grid; sub-samples if
    over budget.
    """
    raise NotImplementedError("week 2: see DESIGN.md §3.4")
