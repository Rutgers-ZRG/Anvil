"""Common base class for DFT labeling engines.

An engine owns everything that is code-specific about turning an ASE `Atoms`
object into a labeled `Atoms` (energy / forces / stress):

    write_inputs()  — one struct dir per structure, engine-specific input files
    submit()        — Slurm submission of those dirs (throttled, idempotent)
    run_local()     — run one struct dir here and now (tests, laptops, inside
                      an allocation); Slurm scripts call the same code path
    is_converged()  — cheap check used for idempotent resubmission
    collect()       — parse outputs, sanity-check, return labeled Atoms

Engines share the on-disk conventions so the orchestrator never has to know
which one is in use:

    <struct_dir>/.anvil_meta.json    sample_kind / pool / engine + options
    <struct_dir>/anvil_result.xyz    extxyz with a SinglePointCalculator
                                     (engines that run through ASE; the VASP
                                     engine reads vasprun.xml instead)

See DESIGN.md §4.3.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from anvil.hpc.clusters import ClusterConfig
from anvil.hpc.slurm import SlurmJobSpec, submit_throttled

META_FILE = ".anvil_meta.json"
RESULT_FILE = "anvil_result.xyz"


class EngineError(RuntimeError):
    """Engine misconfiguration or unusable backend (missing binary, module...)."""


class DFTEngine(ABC):
    """Base class for all labeling backends.

    Subclasses must set `name` and implement `write_inputs`, `is_converged`,
    `_read_result` and `_job_body`.
    """

    name: str = "base"

    def __init__(
        self,
        *,
        cluster: Optional[ClusterConfig] = None,
        options: Optional[dict] = None,
        kspacing: float = 0.25,
    ) -> None:
        self.cluster = cluster
        self.options = dict(options or {})
        self.kspacing = kspacing

    # ---------------------------------------------------------------- inputs

    @abstractmethod
    def write_inputs(
        self,
        atoms: Atoms,
        out_dir: str | Path,
        *,
        sample_kind: str = "",
        pool: str = "",
    ) -> Path:
        """Write all input files for one structure into `out_dir`."""

    def write_meta(
        self,
        out_dir: str | Path,
        *,
        sample_kind: str = "",
        pool: str = "",
        extra: Optional[dict] = None,
    ) -> Path:
        """Write .anvil_meta.json — provenance + what run_local needs later."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "engine": self.name,
            "sample_kind": sample_kind,
            "pool": pool,
            "options": self.options,
        }
        if extra:
            meta.update(extra)
        path = out / META_FILE
        path.write_text(json.dumps(meta, indent=2))
        return path

    @staticmethod
    def read_meta(struct_dir: str | Path) -> dict:
        path = Path(struct_dir) / META_FILE
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    # ------------------------------------------------------------ submission

    @abstractmethod
    def _job_body(self, struct_dir: Path) -> str:
        """Shell body of the Slurm script that labels `struct_dir`."""

    def _job_spec(self, struct_dir: Path) -> SlurmJobSpec:
        """Slurm resources for one structure, from `options['slurm']`."""
        sl = dict(self.options.get("slurm") or {})
        partition = sl.get("partition") or (
            self.cluster.cpu_partition if self.cluster else "main"
        )
        return SlurmJobSpec(
            name=sl.get("job_name") or struct_dir.name,
            partition=partition,
            n_tasks=int(sl.get("ntasks", 32)),
            mem=sl.get("mem", "64G"),
            time=sl.get("time", "03:00:00"),
            output=str(struct_dir / "slurm.out"),
            error=str(struct_dir / "slurm.err"),
            chdir=str(struct_dir) if (self.cluster and self.cluster.requires_chdir) else "",
        )

    def make_job_script(self, struct_dir: str | Path) -> tuple[Path, SlurmJobSpec]:
        """Write sbp.sh in `struct_dir`; return (path, spec)."""
        sd = Path(struct_dir)
        sd.mkdir(parents=True, exist_ok=True)
        spec = self._job_spec(sd)
        script = sd / "sbp.sh"
        script.write_text(spec.render(self._job_body(sd)))
        script.chmod(0o755)
        return script, spec

    def _module_lines(self) -> list[str]:
        """`module load` lines from options, else the cluster's intel module."""
        mods = self.options.get("slurm", {}).get("module_loads")
        if mods is None:
            mods = [self.cluster.intel_module] if self.cluster else []
        return [f"module load {m}" for m in mods]

    def _conda_lines(self) -> list[str]:
        """Activate the DFT conda env when the cluster/options define one."""
        env = self.options.get("conda_env") or (
            getattr(self.cluster, "conda_env_dft", "") if self.cluster else ""
        )
        if not env:
            return []
        root = self.options.get("conda_root") or (
            self.cluster.conda_root if self.cluster else ""
        )
        lines = []
        if root:
            lines.append(f"source {root}/etc/profile.d/conda.sh")
        lines.append(f"conda activate {env}")
        return lines

    def submit(
        self,
        struct_dirs: list[str | Path],
        *,
        user: Optional[str] = None,
        max_concurrent: int = 400,
        ssh_host: Optional[str] = None,
    ) -> list[str]:
        """Submit one job per struct dir, throttled below the queue cap.

        Idempotent: dirs that already hold a converged result are skipped and
        get job_id "DONE".
        """
        job_ids: list[str] = []
        pending_scripts: list[Path] = []
        pending_indices: list[int] = []

        for i, sd in enumerate(struct_dirs):
            if self.is_converged(sd):
                job_ids.append("DONE")
                continue
            script, _ = self.make_job_script(sd)
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

    # -------------------------------------------------------------- labeling

    def run_local(self, struct_dir: str | Path) -> Atoms:
        """Label one struct dir in this process and write anvil_result.xyz.

        Engines that only run through a batch binary (VASP) do not implement
        this; the Slurm path is the only way to run them.
        """
        raise NotImplementedError(
            f"Engine {self.name!r} cannot run in-process; submit it to Slurm."
        )

    def write_result(self, struct_dir: str | Path, atoms: Atoms) -> Path:
        """Write a labeled Atoms as anvil_result.xyz (single extxyz frame)."""
        from ase.io import write as ase_write

        out = Path(struct_dir) / RESULT_FILE
        ase_write(str(out), atoms, format="extxyz")
        return out

    # ------------------------------------------------------------ collection

    @abstractmethod
    def is_converged(self, struct_dir: str | Path) -> bool:
        """True if `struct_dir` holds a finished, converged calculation."""

    @abstractmethod
    def _read_result(self, struct_dir: Path) -> Atoms:
        """Parse the engine's output into an Atoms with energy/forces/stress."""

    def collect(
        self,
        struct_dirs: list[str | Path],
        *,
        drop_unconverged: bool = True,
        force_max_threshold_eV_A: float = 100.0,
    ) -> tuple[list[Atoms], list[str]]:
        """Collect labeled atoms + the list of dirs that failed.

        Applies the same sanity checks for every engine: convergence, finite
        energy, |F| below threshold (catches imploded cells), and ASE stress
        convention tagging (compression-negative — see DESIGN.md §10).
        """
        out_atoms: list[Atoms] = []
        failed: list[str] = []
        for sd in struct_dirs:
            sd = Path(sd)
            if drop_unconverged and not self.is_converged(sd):
                failed.append(str(sd))
                continue
            try:
                a = self._read_result(sd)
                e = float(a.get_potential_energy())
                f = np.asarray(a.get_forces(), dtype=float)
                try:
                    s = np.asarray(a.get_stress(), dtype=float)
                except Exception:
                    s = None  # engine/calculator without stress support
                if not np.isfinite(e) or not np.all(np.isfinite(f)):
                    failed.append(str(sd))
                    continue
                if np.abs(f).max() > force_max_threshold_eV_A:
                    failed.append(str(sd))
                    continue
                results = {"energy": e, "forces": f}
                if s is not None:
                    results["stress"] = s
                a.calc = SinglePointCalculator(a, **results)
                a.info["total_energy"] = e
                a.info["stress_convention"] = "ase"  # compression-negative
                a.info["dft_engine"] = self.name
                meta = self.read_meta(sd)
                for k in ("sample_kind", "pool"):
                    if meta.get(k):
                        a.info[k] = meta[k]
                out_atoms.append(a)
            except Exception:
                failed.append(str(sd))
        return out_atoms, failed
