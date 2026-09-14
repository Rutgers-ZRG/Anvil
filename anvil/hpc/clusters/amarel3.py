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
    pw_bin="/home/lz432/apps/q-e-qe-7.2/bin/pw.x",     # Quantum ESPRESSO 7.2
    qe_modules=("intel/17.0.2",),
    qe_pseudo_root="/home/mw1134/projects/qe_potential",
    # amarel3 offers intel 16.0.3 / 17.0.2 / 18.0.5 — there is no 17.0.4 here,
    # so `module load intel/17.0.4` silently failed and jobs ran without an MPI
    # environment (`which mpirun` finds nothing on the compute nodes; only srun
    # exists). Verified: job 61310969 (17.0.4 + mpirun) SIGSEGV'd in MPI_Init,
    # job 61310970 (17.0.2 + srun) converged in 5 s.
    intel_module="intel/17.0.2",
    mpi_launcher="srun --mpi=pmi2",
    qe_launcher="srun --mpi=pmi2",
    scratch_root="/scratch/lz432",
    requires_chdir=True,                          # default chdir is /cache/home
)
