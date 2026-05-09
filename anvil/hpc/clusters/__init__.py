"""Cluster registry. v1: amareln, amarel3.

See DESIGN.md §4.10, §9 decision #3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClusterConfig:
    name: str
    hostname: str
    scheduler: str = "slurm"
    queue_cap: int = 500                          # silent truncation above this
    gpu_partition: str = "gpu"
    cpu_partition: str = "main"
    conda_root: str = "/home/lz432/miniconda3"
    conda_env_train: str = "nequip"
    foundation_path: str = ""
    vasp_bin: str = "/home/lz432/apps/vasp.6.4.2/bin/vasp_std"
    potcar_root: str = "/home/lz432/apps/PBE64"
    intel_module: str = "intel/17.0.4"
    scratch_root: str = "/scratch/lz432"
    requires_chdir: bool = False                  # amarel3: True


def cluster_for(hostname: str) -> ClusterConfig:
    """Auto-detect cluster from hostname. Raises if unrecognized."""
    raise NotImplementedError("week 1: see anvil.hpc.clusters.{amareln,amarel3}")
