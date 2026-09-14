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
    # Quantum ESPRESSO 7.2 (group build). Note it is compiled against
    # intel/17.0.2, not the intel/17.0.4 the VASP jobs load, and the working
    # QE jobs on this cluster launch with `srun --mpi=pmi2` rather than mpirun.
    pw_bin="/home/lz432/apps/q-e-qe-7.2/bin/pw.x",
    qe_launcher="srun --mpi=pmi2",
    qe_modules=("intel/17.0.2",),
    qe_pseudo_root="/home/mw1134/projects/qe_potential",
    intel_module="intel/17.0.4",
    scratch_root="/scratch/lz432",
    requires_chdir=True,                          # default chdir is /cache/home
)
