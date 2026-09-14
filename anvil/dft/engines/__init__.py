"""Pluggable DFT labeling engines.

Anvil's active-learning loop only ever needs four things from a DFT code:
write inputs, submit, check convergence, collect labeled Atoms. `DFTEngine`
is that contract, so VASP is one option among several rather than the only
backend:

    "vasp"  — VASP + Slurm (v1 default, unchanged behaviour)
    "qe"    — Quantum ESPRESSO through QEpy (in-process) or pw.x (batch)
    "ase"   — any ASE calculator (GPAW, CP2K, Abinit, EMT for smoke tests...)

Select one in the user yaml::

    dft:
      engine: qe
      kspacing: 0.25
      engine_options: {...}       # engine-specific block

See DESIGN.md §4.3.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from anvil.dft.engines.base import DFTEngine, EngineError
from anvil.hpc.clusters import ClusterConfig

if TYPE_CHECKING:  # pragma: no cover
    from anvil.config import DFTConfig

__all__ = [
    "DFTEngine",
    "EngineError",
    "ENGINES",
    "get_engine_class",
    "make_engine",
    "engine_from_struct_dir",
]


def _vasp_engine():
    from anvil.dft.engines.vasp_engine import VaspEngine

    return VaspEngine


def _qe_engine():
    from anvil.dft.engines.qe import QEEngine

    return QEEngine


def _ase_engine():
    from anvil.dft.engines.ase_calc import ASECalculatorEngine

    return ASECalculatorEngine


# Lazy loaders — importing an engine must not require its backend installed.
ENGINES: dict[str, callable] = {
    "vasp": _vasp_engine,
    "qe": _qe_engine,
    "espresso": _qe_engine,      # alias
    "qepy": _qe_engine,          # alias
    "ase": _ase_engine,
}


def get_engine_class(name: str):
    """Resolve an engine name to its class."""
    key = str(name).lower()
    if key not in ENGINES:
        raise EngineError(
            f"Unknown DFT engine {name!r}. Known: {sorted(set(ENGINES))}"
        )
    return ENGINES[key]()


def make_engine(
    dft_cfg: "DFTConfig",
    *,
    cluster: Optional[ClusterConfig] = None,
    cat_potcar: bool = True,
) -> DFTEngine:
    """Build the engine described by a `DFTConfig`.

    `cat_potcar` only affects the VASP engine (skip POTCAR concatenation when
    the inputs are staged on a remote cluster over SSH).
    """
    cls = get_engine_class(dft_cfg.engine)
    options = dict(getattr(dft_cfg, "engine_options", None) or {})
    if cls.name == "vasp":
        return cls(
            cluster=cluster,
            options=options,
            kspacing=dft_cfg.kspacing,
            functional=dft_cfg.functional,
            encut=dft_cfg.encut,
            extra_incar=dft_cfg.extra_incar or None,
            cat_potcar=cat_potcar,
        )
    return cls(cluster=cluster, options=options, kspacing=dft_cfg.kspacing)


def engine_from_struct_dir(
    struct_dir: str | Path,
    *,
    cluster: Optional[ClusterConfig] = None,
) -> DFTEngine:
    """Rebuild the engine that wrote `struct_dir`, from its .anvil_meta.json.

    Used by `anvil.dft.engines._run` on the compute node, where the user yaml
    is not necessarily at hand.
    """
    meta = DFTEngine.read_meta(struct_dir)
    name = meta.get("engine")
    if not name:
        raise EngineError(
            f"{struct_dir} has no .anvil_meta.json with an `engine` key — "
            f"was it written by an Anvil engine?"
        )
    cls = get_engine_class(name)
    options = meta.get("options") or {}
    if cls.name == "vasp":
        return cls(cluster=cluster, options=options)
    return cls(cluster=cluster, options=options)
