"""Cluster registry. v1: amareln, amarel3.

See DESIGN.md §4.10, §9 decision #3.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterConfig:
    name: str
    hostname: str
    scheduler: str = "slurm"
    queue_cap: int = 500                           # silent truncation above this
    gpu_partition: str = "gpu"
    cpu_partition: str = "main"
    conda_root: str = "/home/lz432/miniconda3"
    conda_env_train: str = "nequip"
    foundation_path: str = ""
    vasp_bin: str = "/home/lz432/apps/vasp.6.4.2/bin/vasp_std"
    potcar_root: str = "/home/lz432/apps/PBE64"
    intel_module: str = "intel/17.0.4"
    scratch_root: str = "/scratch/lz432"
    requires_chdir: bool = False                   # amarel3: True


def get_cluster(name: str) -> ClusterConfig:
    """Look up a registered cluster by canonical name."""
    from anvil.hpc.clusters.amareln import CLUSTER as AMARELN
    from anvil.hpc.clusters.amarel3 import CLUSTER as AMAREL3

    registry = {
        "amareln": AMARELN,
        "amarel3": AMAREL3,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown cluster {name!r}. Known: {list(registry)}"
        )
    return registry[name]


def cluster_for_host(hostname: str | None = None) -> ClusterConfig:
    """Auto-detect cluster from hostname (defaults to socket.gethostname())."""
    h = hostname or socket.gethostname()
    h_lower = h.lower()
    if "amareln" in h_lower or h_lower.startswith("haln"):
        return get_cluster("amareln")
    if "amarel3" in h_lower or h_lower.startswith("gpun") or h_lower.startswith("slepner"):
        return get_cluster("amarel3")
    raise ValueError(
        f"Hostname {h!r} not recognized as amareln or amarel3. "
        f"For local development, pass cluster=... explicitly."
    )
