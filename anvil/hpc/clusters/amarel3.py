"""amarel3 (newer ampere/L40S partition, GLIBC 2.34, requires explicit --chdir)."""

from __future__ import annotations

from anvil.hpc.clusters import ClusterConfig


CLUSTER = ClusterConfig(
    name="amarel3",
    hostname="amarel3.hpc.rutgers.edu",
    scheduler="slurm",
    queue_cap=500,
    gpu_partition="gpu",
    cpu_partition="main",
    conda_root="/home/lz432/miniconda3",
    conda_env_train="nequip",
    foundation_path=(
        "/scratch/lz432/allegro_finetune/allegro-oam-l-foundation.nequip.pth"
    ),
    vasp_bin="/home/lz432/apps/vasp.6.4.2/bin/vasp_std",
    potcar_root="/home/lz432/apps/PBE64",
    intel_module="intel/17.0.4",
    scratch_root="/scratch/lz432",
    requires_chdir=True,                          # default chdir is /cache/home
)
