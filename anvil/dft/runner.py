"""Back-compat wrappers around the VASP engine.

The v1 helpers below predate the `DFTEngine` abstraction and are kept so that
existing scripts keep working. New code should go through
`anvil.dft.engines.make_engine(cfg.dft, cluster=...)`, which returns whichever
backend the user selected (VASP, Quantum ESPRESSO, or any ASE calculator).

See anvil/dft/engines/ and DESIGN.md §4.3, §10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ase import Atoms

from anvil.dft.engines.vasp_engine import VaspEngine
from anvil.hpc.clusters import ClusterConfig
from anvil.hpc.slurm import SlurmJobSpec


def _engine(
    cluster: Optional[ClusterConfig] = None,
    functional_yaml: Optional[dict] = None,
    *,
    job_name: str | None = None,
) -> VaspEngine:
    fy = functional_yaml or {}
    options: dict = {"slurm": dict(fy.get("slurm") or {})}
    if job_name:
        options["slurm"]["job_name"] = job_name
    return VaspEngine(cluster=cluster, options=options, functional_yaml=fy)


def make_vasp_slurm_script(
    struct_dir: str | Path,
    cluster: ClusterConfig,
    functional_yaml: dict,
    *,
    job_name: str | None = None,
) -> tuple[Path, SlurmJobSpec]:
    """Write sbp.sh in struct_dir, return (path, spec)."""
    return _engine(cluster, functional_yaml, job_name=job_name).make_job_script(struct_dir)


def submit_vasp_array(
    struct_dirs: list[str | Path],
    cluster: ClusterConfig,
    functional_yaml: dict,
    *,
    user: str | None = None,
    max_concurrent: int = 400,
    ssh_host: str | None = None,
) -> list[str]:
    """Write sbp.sh per dir and submit, throttled; converged dirs get "DONE"."""
    return _engine(cluster, functional_yaml).submit(
        struct_dirs, user=user, max_concurrent=max_concurrent, ssh_host=ssh_host,
    )


def is_converged(struct_dir: str | Path) -> bool:
    """True if OUTCAR contains 'Total CPU time' or 'reached required accuracy'."""
    return _engine().is_converged(struct_dir)


def collect_vasp_outputs(
    struct_dirs: list[str | Path],
    *,
    drop_unconverged: bool = True,
    force_max_threshold_eV_A: float = 100.0,
) -> tuple[list[Atoms], list[str]]:
    """Collect labeled atoms from vasprun.xml + list of failed dirs."""
    return _engine().collect(
        struct_dirs,
        drop_unconverged=drop_unconverged,
        force_max_threshold_eV_A=force_max_threshold_eV_A,
    )
