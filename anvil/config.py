"""User-yaml configuration schema + validation.

See DESIGN.md §2 for the user input spec and default values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SeedConfig:
    """Seed-structure source. Exactly one of from_mp / from_local must be set."""
    from_mp: Optional[list[str]] = None
    from_local: Optional[list[str]] = None


@dataclass
class DFTConfig:
    functional: str = "pbe"
    encut: int = 600
    kspacing: float = 0.25
    extra_incar: dict = field(default_factory=dict)


@dataclass
class RegimeConfig:
    pressures_gpa: list[float] = field(default_factory=lambda: [0])
    temperatures_k: list[float] = field(default_factory=lambda: [300])


@dataclass
class BudgetConfig:
    max_dft_calls: int = 800
    max_walltime_h: int = 48


@dataclass
class ReformRelaxConfig:
    mode: str = "stability"  # "stability" (mode a) | "discovery" (mode b)
    pyxtal_n_per_round: int = 20
    perturb_rattle: float = 0.3
    perturb_cell: float = 0.05
    relax_max_steps: int = 200
    relax_fmax: float = 0.05
    sample_intermediates: int = 2
    spglib_filter: bool = True
    pallas_pathway_xyz: Optional[str] = None  # v1 hook; v2 full pipeline


@dataclass
class AnchorConfig:
    strain_factors: list[float] = field(
        default_factory=lambda: [0.95, 0.97, 1.03, 1.05]
    )
    md_steps: int = 100
    md_snapshots: int = 5


@dataclass
class EntropyMDConfig:
    """Pool A — per-regime entropy MD.

    Default (T × k_factor × mode) grid; pressures and phases come from
    RegimeConfig and seeds.
    """
    k_factors: list[float] = field(default_factory=lambda: [0, 2, 5])
    modes: list[str] = field(default_factory=lambda: ["per_atom", "per_config"])
    md_steps: int = 1000
    fp_cutoff: float = 5.0
    fp_natx: int = 50
    regularization: float = 1e-3
    min_entropy_gain: float = 0.01


@dataclass
class GenerationQuotas:
    """Per-round DFT call quota across the three pools (sum = per_round_dft)."""
    pool_a_entropy: int = 50
    pool_b_reform: int = 20
    pool_c_anchor: int = 10


@dataclass
class ValidationConfig:
    tier: int = 2
    thresholds: dict = field(
        default_factory=lambda: {"E_MAE": 10.0, "F_MAE": 100.0, "S_MAE": 10.0}
    )


@dataclass
class TrainingConfig:
    foundation_model: str = "allegro-oam-l-foundation"
    ensemble_size: int = 3
    bootstrap_size: int = 100
    per_round_dft: int = 60
    loss_weights: str = "1:1:0.01"  # E:F:S
    learning_rate: float = 5e-5
    patience: int = 30
    max_epochs: int = 200
    compile_target: str = "torchscript-ase"


@dataclass
class AnvilConfig:
    """Top-level user config. Loaded from yaml; defaults in DESIGN.md §2."""

    system: str
    elements: list[str]
    seeds: SeedConfig
    dft: DFTConfig
    regime: RegimeConfig
    budget: BudgetConfig

    # All optional with defaults
    entropy_md: EntropyMDConfig = field(default_factory=EntropyMDConfig)
    reform_relax: ReformRelaxConfig = field(default_factory=ReformRelaxConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    generation_quotas: GenerationQuotas = field(default_factory=GenerationQuotas)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> AnvilConfig:
    """Load and validate a user yaml.

    Raises ConfigError with a helpful message on schema mismatch.
    """
    raise NotImplementedError("week 1: see DESIGN.md §2 + JSON schema")


def validate_config(cfg: AnvilConfig) -> None:
    """Schema + cross-field consistency checks.

    - Foundation model covers all elements (via FoundationRegistry).
    - DFT functional has an INCAR template under configs/functionals/.
    - Pool quotas sum ≤ per_round_dft.
    - Validation tier ∈ {1, 2, 3}.
    - PALLAS pathway xyz exists (if specified).
    """
    raise NotImplementedError("week 1: see DESIGN.md §4.4 element coverage")


class ConfigError(ValueError):
    """Raised on schema validation failure with user-friendly diagnostics."""
