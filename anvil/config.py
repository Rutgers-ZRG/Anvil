"""User-yaml configuration schema + loader + validation.

See DESIGN.md §2 for the user input spec and default values.
"""

from __future__ import annotations

import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class SeedConfig:
    """Seed-structure source. Exactly one of from_mp / from_local must be set."""
    from_mp: Optional[list[str]] = None
    from_local: Optional[list[str]] = None


@dataclass
class DFTConfig:
    """Labeling backend + its settings.

    `engine` selects the code that produces energies/forces/stresses:
    "vasp" (default), "qe" (Quantum ESPRESSO via QEpy or pw.x), or "ase"
    (any ASE calculator). `functional`/`encut`/`extra_incar` are VASP-side
    knobs; everything engine-specific lives in `engine_options`.
    See anvil/dft/engines/ and DESIGN.md §4.3.
    """
    engine: str = "vasp"
    functional: str = "pbe"
    encut: int = 600
    kspacing: float = 0.25
    extra_incar: dict = field(default_factory=dict)
    engine_options: dict = field(default_factory=dict)


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
    mode: str = "stability"               # "stability" (mode a) | "discovery" (mode b)
    pyxtal_n_per_round: int = 20
    perturb_rattle: float = 0.3
    perturb_cell: float = 0.05
    relax_max_steps: int = 200
    relax_fmax: float = 0.05
    sample_intermediates: int = 2
    spglib_filter: bool = True
    pallas_pathway_xyz: Optional[str] = None


@dataclass
class AnchorConfig:
    strain_factors: list[float] = field(
        default_factory=lambda: [0.95, 0.97, 1.03, 1.05]
    )
    md_steps: int = 100
    md_snapshots: int = 5


@dataclass
class EntropyMDConfig:
    k_factors: list[float] = field(default_factory=lambda: [0, 2, 5])
    modes: list[str] = field(default_factory=lambda: ["per_atom", "per_config"])
    md_steps: int = 1000
    fp_cutoff: float = 5.0
    fp_natx: int = 50
    regularization: float = 1e-3
    min_entropy_gain: float = 0.01


@dataclass
class GenerationQuotas:
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
    per_round_dft: int = 80                # matches DESIGN.md §3.5 default sum of A/B/C quotas
    loss_weights: str = "1:1:0.01"
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

    entropy_md: EntropyMDConfig = field(default_factory=EntropyMDConfig)
    reform_relax: ReformRelaxConfig = field(default_factory=ReformRelaxConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    generation_quotas: GenerationQuotas = field(default_factory=GenerationQuotas)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


class ConfigError(ValueError):
    """Schema validation failure with user-friendly diagnostics."""


def _coerce(cls, value):
    """Recursively coerce a dict into a dataclass instance.

    Resolves PEP 563 string annotations via typing.get_type_hints so nested
    dataclasses are recognized even with `from __future__ import annotations`.
    For unknown keys, raises ConfigError (helps the user catch typos).
    """
    if value is None:
        return cls()
    if not isinstance(value, dict):
        raise ConfigError(f"Expected dict for {cls.__name__}, got {type(value).__name__}")
    known = {f.name: f for f in fields(cls)}
    extras = set(value) - set(known)
    if extras:
        raise ConfigError(
            f"Unknown keys for {cls.__name__}: {sorted(extras)}. "
            f"Allowed: {sorted(known)}"
        )
    # Resolve string annotations to real classes
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in value:
            continue
        v = value[name]
        resolved = hints.get(name, known[name].type)
        # Strip Optional / Union for dataclass detection
        origin = typing.get_origin(resolved)
        if origin is typing.Union:
            args = [a for a in typing.get_args(resolved) if a is not type(None)]
            if len(args) == 1:
                resolved = args[0]
        if isinstance(resolved, type) and is_dataclass(resolved):
            kwargs[name] = _coerce(resolved, v)
        else:
            kwargs[name] = v
    return cls(**kwargs)


def load_config(path: str | Path) -> AnvilConfig:
    """Load and validate a user yaml.

    Defaults are applied via the dataclass field defaults; the user yaml
    only needs to set the required fields (system, elements, seeds, dft,
    regime, budget) plus any overrides.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level yaml must be a mapping, got {type(raw).__name__}")

    # Required top-level fields
    required = {"system", "elements", "seeds", "dft", "regime", "budget"}
    missing = required - set(raw)
    if missing:
        raise ConfigError(f"Missing required top-level keys: {sorted(missing)}")

    cfg = _coerce(AnvilConfig, raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: AnvilConfig) -> None:
    """Schema + cross-field consistency checks."""
    if not cfg.system or not isinstance(cfg.system, str):
        raise ConfigError("`system` must be a non-empty string")
    if not cfg.elements:
        raise ConfigError("`elements` must be a non-empty list")

    s = cfg.seeds
    if not (s.from_mp or s.from_local):
        raise ConfigError("`seeds` must specify exactly one of from_mp / from_local")
    if s.from_mp and s.from_local:
        raise ConfigError("`seeds` cannot specify both from_mp and from_local")

    _validate_dft(cfg)

    if cfg.validation.tier not in (1, 2, 3):
        raise ConfigError(f"`validation.tier` must be 1, 2, or 3 (got {cfg.validation.tier})")

    if cfg.reform_relax.mode not in ("stability", "discovery"):
        raise ConfigError(
            f"`reform_relax.mode` must be 'stability' or 'discovery' "
            f"(got {cfg.reform_relax.mode!r})"
        )

    if cfg.budget.max_dft_calls <= 0:
        raise ConfigError("`budget.max_dft_calls` must be > 0")

    # Quotas should not exceed per_round_dft
    q = cfg.generation_quotas
    total_quota = q.pool_a_entropy + q.pool_b_reform + q.pool_c_anchor
    if total_quota > cfg.training.per_round_dft + 5:    # small slop for rounding
        raise ConfigError(
            f"Sum of generation quotas ({total_quota}) exceeds per_round_dft "
            f"({cfg.training.per_round_dft})"
        )


def _validate_dft(cfg: AnvilConfig) -> None:
    """Check `dft.engine` and the engine-specific options block."""
    from anvil.dft.engines import ENGINES

    engine = str(cfg.dft.engine).lower()
    if engine not in ENGINES:
        raise ConfigError(
            f"Unknown `dft.engine` {cfg.dft.engine!r}. Known: {sorted(set(ENGINES))}"
        )
    opts = cfg.dft.engine_options or {}

    if engine in ("qe", "espresso", "qepy"):
        mode = str(opts.get("mode", "qepy")).lower()
        if mode not in ("qepy", "pwx"):
            raise ConfigError(
                f"`dft.engine_options.mode` must be 'qepy' or 'pwx' (got {mode!r})"
            )
        pseudos = opts.get("pseudopotentials") or {}
        missing = [el for el in cfg.elements if el not in pseudos]
        if missing:
            raise ConfigError(
                f"`dft.engine_options.pseudopotentials` is missing entries for "
                f"{missing} (engine 'qe' needs one UPF per element)"
            )
        if not opts.get("pseudo_dir"):
            raise ConfigError(
                "`dft.engine_options.pseudo_dir` is required for engine 'qe'"
            )
    elif engine == "ase":
        if not opts.get("calculator"):
            raise ConfigError(
                "`dft.engine_options.calculator` is required for engine 'ase' "
                "(e.g. 'gpaw.GPAW', 'ase.calculators.cp2k.CP2K', or 'emt')"
            )


def to_canonical_json(cfg: AnvilConfig) -> str:
    """Serialize cfg to canonical JSON (sorted keys, recursive dataclass→dict).

    Used for content-hash provenance.
    """
    def _walk(obj):
        if is_dataclass(obj):
            return {f.name: _walk(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_walk(v) for v in obj]
        return obj
    return json.dumps(_walk(cfg), sort_keys=True, indent=2)
