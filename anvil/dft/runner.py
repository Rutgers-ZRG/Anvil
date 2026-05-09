"""Slurm submission, polling, and output collection for DFT jobs.

See DESIGN.md §4.3, §10.

Key features:
  - Throttled submit (queue-cap aware).
  - Idempotent: re-running a struct dir checks for existing converged OUTCAR
    before re-submitting.
  - Output validation: convergence + |F| sanity (drop unphysical |F| > 100 eV/Å)
    BEFORE adding to training pool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from anvil.hpc.clusters import ClusterConfig
from anvil.hpc.slurm import SlurmJobSpec, submit_throttled


def make_vasp_slurm_script(
    struct_dir: str | Path,
    cluster: ClusterConfig,
    functional_yaml: dict,
    *,
    job_name: str | None = None,
) -> tuple[Path, SlurmJobSpec]:
    """Write sbp.sh in struct_dir, return (path, spec).

    The script uses the cluster's intel module + vasp_bin and routes mpirun
    via srun --mpi=pmi2.
    """
    sd = Path(struct_dir)
    sd.mkdir(parents=True, exist_ok=True)

    sl = functional_yaml.get("slurm", {})
    spec = SlurmJobSpec(
        name=job_name or sd.name,
        partition=sl.get("partition", cluster.cpu_partition),
        n_tasks=int(sl.get("ntasks", 32)),
        mem=sl.get("mem", "64G"),
        time=sl.get("time", "03:00:00"),
        output=str(sd / "slurm.out"),
        error=str(sd / "slurm.err"),
        chdir=str(sd) if cluster.requires_chdir else "",
    )
    module_loads = sl.get("module_loads") or [cluster.intel_module]
    body_lines = []
    for m in module_loads:
        body_lines.append(f"module load {m}")
    body_lines.append(f"cd {sd}")
    body_lines.append(
        f"mpirun -n {spec.n_tasks} {cluster.vasp_bin} > vasp.log 2>&1"
    )
    body = "\n".join(body_lines)
    script = sd / "sbp.sh"
    script.write_text(spec.render(body))
    script.chmod(0o755)
    return script, spec


def submit_vasp_array(
    struct_dirs: list[str | Path],
    cluster: ClusterConfig,
    functional_yaml: dict,
    *,
    user: str | None = None,
    max_concurrent: int = 400,
    ssh_host: str | None = None,
) -> list[str]:
    """For each struct_dir: write sbp.sh, then submit via throttled Slurm.

    Idempotent: if a dir already has a converged OUTCAR, skip submission and
    return job_id="DONE".
    """
    job_ids: list[str] = []
    pending_scripts: list[Path] = []
    pending_indices: list[int] = []

    for i, sd in enumerate(struct_dirs):
        if is_converged(sd):
            job_ids.append("DONE")
            continue
        script, _ = make_vasp_slurm_script(sd, cluster, functional_yaml)
        pending_scripts.append(script)
        pending_indices.append(i)
        job_ids.append("")  # placeholder

    if not pending_scripts:
        return job_ids

    submitted = submit_throttled(
        pending_scripts,
        user=user,
        max_concurrent=max_concurrent,
        ssh_host=ssh_host,
    )
    for idx, jid in zip(pending_indices, submitted):
        job_ids[idx] = jid

    return job_ids


def is_converged(struct_dir: str | Path) -> bool:
    """True if OUTCAR contains 'Total CPU time' or 'reached required accuracy'."""
    outcar = Path(struct_dir) / "OUTCAR"
    if not outcar.exists():
        return False
    try:
        text = outcar.read_text(errors="ignore")
    except OSError:
        return False
    return ("Total CPU time" in text) or ("reached required accuracy" in text)


def collect_vasp_outputs(
    struct_dirs: list[str | Path],
    *,
    drop_unconverged: bool = True,
    force_max_threshold_eV_A: float = 100.0,
) -> tuple[list[Atoms], list[str]]:
    """Collect labeled atoms + list of failed dirs.

    Reads vasprun.xml (which gives ASE-convention compression-negative stress),
    attaches SinglePointCalculator, applies sanity checks.
    """
    import json as _json
    import numpy as np
    from ase.io import read as ase_read

    out_atoms: list[Atoms] = []
    failed: list[str] = []
    for sd in struct_dirs:
        sd = Path(sd)
        if drop_unconverged and not is_converged(sd):
            failed.append(str(sd))
            continue
        vp = sd / "vasprun.xml"
        if not vp.exists():
            failed.append(str(sd))
            continue
        try:
            a = ase_read(str(vp), format="vasp-xml")
            e = float(a.get_potential_energy())
            f = np.array(a.get_forces())
            s = np.array(a.get_stress())
            # Sanity check on forces — catch broken cells silently
            if np.abs(f).max() > force_max_threshold_eV_A:
                failed.append(str(sd))
                continue
            a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=s)
            a.info["total_energy"] = e
            a.info["stress_convention"] = "ase"  # compression-negative
            # Restore Anvil metadata if present
            meta_path = sd / ".anvil_meta.json"
            if meta_path.exists():
                meta = _json.loads(meta_path.read_text())
                for k in ("sample_kind", "pool"):
                    if k in meta:
                        a.info[k] = meta[k]
            out_atoms.append(a)
        except Exception:
            failed.append(str(sd))
    return out_atoms, failed
