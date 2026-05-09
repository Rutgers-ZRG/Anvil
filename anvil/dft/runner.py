"""Slurm submission, polling, and output collection for DFT jobs.

See DESIGN.md §4.3, §4.10 (cluster registry), §10 (operational lessons).

Key features:
  - Throttled submit (queue-cap aware, retry with backoff). Lessons learned:
    silent truncation when amareln 500-job cap exceeded.
  - Idempotency: re-running a struct dir checks for existing converged OUTCAR
    before re-submitting.
  - Output validation: convergence + |F| sanity (drop unphysical |F| > 100 eV/Å)
    BEFORE adding to training pool. Lessons learned: NaCl B2 hot-MD timeouts
    that silently failed if not validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ase import Atoms


def submit_vasp_array(
    struct_dirs: list[str],
    cluster: "ClusterConfig",  # noqa: F821 — anvil.hpc.clusters
    *,
    throttle: bool = True,
    max_concurrent: Optional[int] = None,
) -> list[str]:
    """Submit VASP single-point jobs for each struct dir, return Slurm job IDs."""
    raise NotImplementedError("week 1: see DESIGN.md §4.10 throttled submit")


def poll_vasp_jobs(job_ids: list[str]) -> dict[str, str]:
    """Return {job_id: state} via squeue + sacct."""
    raise NotImplementedError("week 1")


def collect_vasp_outputs(
    struct_dirs: list[str],
    *,
    drop_unconverged: bool = True,
    force_max_threshold_eV_A: float = 100.0,
) -> tuple[list[Atoms], list[str]]:
    """Collect labeled atoms + list of failed dirs.

    Reads vasprun.xml, attaches SinglePointCalculator, applies stress sign
    convention (ASE: compression-negative), filters out unconverged or
    insane-force outputs.
    """
    raise NotImplementedError("week 1: see anvil.dft.validate")
