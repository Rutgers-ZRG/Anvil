"""Slurm submission, polling, and queue-cap-aware throttling.

See DESIGN.md §4.10, §10 (queue-cap truncation lesson).
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SlurmJobSpec:
    name: str
    partition: str
    n_tasks: int = 1
    n_nodes: int = 1
    mem: str = "8G"
    time: str = "01:00:00"
    output: str = "%x_%j.out"
    error: str = "%x_%j.err"
    gres: str = ""              # e.g. "gpu:1"
    chdir: str = ""              # explicit chdir (avoid amarel3 /cache/home issue)
    extra_directives: list[str] | None = None


def submit(
    spec: SlurmJobSpec,
    script_body: str,
    *,
    throttled: bool = True,
    max_concurrent: int = 400,
    backoff_seconds: float = 30.0,
) -> str:
    """Submit one job; return Slurm job ID.

    `throttled=True` blocks while squeue > max_concurrent and retries with
    backoff_seconds delay. Avoids the amareln 500-job silent truncation.
    """
    raise NotImplementedError("week 1")


def submit_array(specs: list[tuple[SlurmJobSpec, str]], **kw) -> list[str]:
    """Submit many jobs efficiently with queue throttling. Returns job IDs."""
    raise NotImplementedError("week 1")


def poll(job_ids: list[str]) -> dict[str, str]:
    """Return {job_id: state}. State ∈ {PENDING, RUNNING, COMPLETED, FAILED, ...}.

    Combines `squeue` (live) and `sacct` (terminal) lookups.
    """
    raise NotImplementedError("week 1")
