"""Quantum ESPRESSO engine, driven either by QEpy or by the `pw.x` binary.

Two modes, selected with `dft.engine_options.mode`:

  "qepy" (default)
      QE runs *in process* through QEpy's ASE calculator
      (`qepy.calculator.QEpyCalculator`). Under Slurm the job body is
      `mpirun -n N python -m anvil.dft.engines._run <struct_dir>`, so the SCF
      still runs on compute nodes under MPI — QEpy just replaces the file
      round-trip, and gives Anvil a Python handle on the QE state for later
      work (v2: per-step forces during MD labeling).

  "pwx"
      Classic batch run: write `pw.in`, `mpirun -n N pw.x -in pw.in > pw.out`,
      parse `pw.out` with ASE. Use this when QEpy is not installed on the
      cluster.

Both modes were checked to agree to SCF noise on rattled Si (dE ~ 2e-5 eV,
dF ~ 2e-7 eV/Å). Note the f90wrap pin in pyproject's [qe] extra: with f90wrap
0.3.x the QEpy wheels raise "0-th dimension must be fixed to 2 but got 4" from
Driver.get_forces() — energy and stress survive, forces do not.

Config sketch::

    dft:
      engine: qe
      kspacing: 0.25
      engine_options:
        mode: qepy
        pseudo_dir: /home/lz432/apps/qe_pseudo
        pseudopotentials:
          Si: Si.pbe-n-kjpaw_psl.1.0.0.UPF
        input_data:
          system:
            ecutwfc: 60
            ecutrho: 480
          electrons:
            conv_thr: 1.0e-8
        slurm: {ntasks: 32, mem: 64G, time: "03:00:00"}

Energies/forces/stresses come back in ASE units (eV, eV/Å, eV/Å³) with the
ASE compression-negative stress convention, exactly like the VASP path.
"""

from __future__ import annotations

import copy
import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from ase import Atoms

from anvil.dft.engines.base import RESULT_FILE, DFTEngine, EngineError
from anvil.dft.vasp import kpoint_mesh
from anvil.hpc.clusters import ClusterConfig

PW_INPUT = "pw.in"
PW_OUTPUT = "pw.out"

# Single-point defaults: forces + stress on, no wavefunction/charge I/O.
DEFAULT_INPUT_DATA: dict = {
    "control": {
        "calculation": "scf",
        "tprnfor": True,
        "tstress": True,
        "disk_io": "none",
        "prefix": "anvil",
    },
    "system": {
        "ecutwfc": 60.0,
        "ecutrho": 480.0,
        "occupations": "smearing",
        "smearing": "gaussian",
        "degauss": 0.01,
    },
    "electrons": {
        "conv_thr": 1.0e-8,
        "electron_maxstep": 200,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins, nested namelists merge per key."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class QEEngine(DFTEngine):
    """Quantum ESPRESSO labeling via QEpy (in-process) or pw.x (batch)."""

    name = "qe"

    def __init__(
        self,
        *,
        cluster: Optional[ClusterConfig] = None,
        options: Optional[dict] = None,
        kspacing: float = 0.25,
    ) -> None:
        super().__init__(cluster=cluster, options=options, kspacing=kspacing)
        self.mode = str(self.options.get("mode", "qepy")).lower()
        if self.mode not in ("qepy", "pwx"):
            raise EngineError(
                f"dft.engine_options.mode must be 'qepy' or 'pwx' (got {self.mode!r})"
            )
        self.pseudopotentials: dict = dict(self.options.get("pseudopotentials") or {})
        self.pseudo_dir = self.options.get("pseudo_dir") or (
            getattr(self.cluster, "qe_pseudo_root", "") if self.cluster else ""
        )
        self.input_data = _deep_merge(
            DEFAULT_INPUT_DATA, self.options.get("input_data") or {}
        )
        if self.pseudo_dir:
            self.input_data["control"]["pseudo_dir"] = str(self.pseudo_dir)

    # ---------------------------------------------------------------- inputs

    def _pseudos_for(self, atoms: Atoms) -> dict[str, str]:
        symbols = sorted(set(atoms.get_chemical_symbols()))
        missing = [s for s in symbols if s not in self.pseudopotentials]
        if missing:
            raise EngineError(
                f"No pseudopotential given for {missing} — set "
                f"dft.engine_options.pseudopotentials for every element."
            )
        return {s: self.pseudopotentials[s] for s in symbols}

    def write_inputs(
        self,
        atoms: Atoms,
        out_dir: str | Path,
        *,
        sample_kind: str = "",
        pool: str = "",
    ) -> Path:
        from ase.io.espresso import write_espresso_in

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        kpts = kpoint_mesh(atoms, self.kspacing)
        input_data = copy.deepcopy(self.input_data)
        input_data.setdefault("control", {})["outdir"] = str(
            self.options.get("outdir", "./tmp")
        )
        with open(out / PW_INPUT, "w") as fd:
            write_espresso_in(
                fd,
                atoms,
                input_data=input_data,
                pseudopotentials=self._pseudos_for(atoms),
                kpts=tuple(kpts),
                koffset=tuple(self.options.get("koffset", (0, 0, 0))),
            )
        self.write_meta(
            out, sample_kind=sample_kind, pool=pool,
            extra={"mode": self.mode, "kpts": kpts},
        )
        return out

    # ------------------------------------------------------------ submission

    def _job_body(self, struct_dir: Path) -> str:
        spec = self._job_spec(struct_dir)
        lines = [*self._module_lines(), *self._conda_lines(), f"cd {struct_dir}"]
        if self.mode == "qepy":
            python_bin = self._python_bin()
            lines.append(
                f"mpirun -n {spec.n_tasks} {python_bin} -m anvil.dft.engines._run "
                f"{shlex.quote(str(struct_dir))} > qepy.log 2>&1"
            )
        else:
            pw_bin = self.options.get("pw_bin") or (
                getattr(self.cluster, "pw_bin", "pw.x") if self.cluster else "pw.x"
            )
            lines.append(
                f"mpirun -n {spec.n_tasks} {pw_bin} -in {PW_INPUT} > {PW_OUTPUT} 2>&1"
            )
        return "\n".join(lines)

    # -------------------------------------------------------------- labeling

    def run_local(self, struct_dir: str | Path) -> Atoms:
        """Run one structure here (inside an allocation, or on a laptop)."""
        sd = Path(struct_dir)
        pwin = sd / PW_INPUT
        if not pwin.exists():
            raise FileNotFoundError(f"No {PW_INPUT} in {sd} — call write_inputs first")
        atoms = self._run_qepy(sd) if self.mode == "qepy" else self._run_pwx(sd)
        if _is_rank_zero():
            self.write_result(sd, atoms)
        return atoms

    def _run_qepy(self, sd: Path) -> Atoms:
        from ase.io import read as ase_read

        try:
            from qepy.calculator import QEpyCalculator
        except ImportError as exc:  # pragma: no cover - depends on cluster env
            raise EngineError(
                "mode='qepy' needs QEpy installed (pip install qepy, or "
                "`pip install anvil-mlip[qe]`). Use mode='pwx' to shell out to "
                "pw.x instead."
            ) from exc

        comm = _mpi_comm()
        cwd = Path.cwd()
        calc = None
        os.chdir(sd)
        try:
            try:
                calc = QEpyCalculator(inputfile=PW_INPUT, comm=comm, logfile=PW_OUTPUT)
            except TypeError:
                # Older QEpy releases do not take `logfile`.
                calc = QEpyCalculator(inputfile=PW_INPUT, comm=comm)
            atoms = ase_read(PW_INPUT, format="espresso-in")
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            try:
                stress = atoms.get_stress()
            except Exception:
                stress = None
            atoms = _detach(atoms, energy, forces, stress)
        finally:
            if calc is not None:
                try:
                    calc.stop()  # release the QE state / free the Fortran side
                except Exception:
                    pass
            os.chdir(cwd)
        return atoms

    def _run_pwx(self, sd: Path) -> Atoms:
        from ase.io import read as ase_read

        pw_bin = self.options.get("pw_bin") or (
            getattr(self.cluster, "pw_bin", "pw.x") if self.cluster else "pw.x"
        )
        ntasks = int(self.options.get("local_ntasks", 1))
        cmd = shlex.split(str(pw_bin)) + ["-in", PW_INPUT]
        if ntasks > 1:
            cmd = ["mpirun", "-n", str(ntasks)] + cmd
        with open(sd / PW_OUTPUT, "w") as out:
            proc = subprocess.run(cmd, cwd=sd, stdout=out, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise EngineError(f"pw.x failed in {sd} (exit {proc.returncode}); see {PW_OUTPUT}")
        return ase_read(str(sd / PW_OUTPUT), format="espresso-out")

    # ------------------------------------------------------------ collection

    def is_converged(self, struct_dir: str | Path) -> bool:
        sd = Path(struct_dir)
        if (sd / RESULT_FILE).exists():
            return True
        pwout = sd / PW_OUTPUT
        if not pwout.exists():
            return False
        try:
            text = pwout.read_text(errors="ignore")
        except OSError:
            return False
        if "convergence NOT achieved" in text:
            return False
        return "JOB DONE." in text and "!    total energy" in text

    def _read_result(self, struct_dir: Path) -> Atoms:
        from ase.io import read as ase_read

        result = struct_dir / RESULT_FILE
        if result.exists():
            return ase_read(str(result), format="extxyz")
        pwout = struct_dir / PW_OUTPUT
        if not pwout.exists():
            raise FileNotFoundError(f"No {RESULT_FILE} or {PW_OUTPUT} in {struct_dir}")
        return ase_read(str(pwout), format="espresso-out")


def _mpi_comm():
    """The MPI communicator to hand QEpy, or None for a serial run."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.COMM_WORLD


def _is_rank_zero() -> bool:
    comm = _mpi_comm()
    return comm is None or comm.rank == 0


def _detach(atoms: Atoms, energy, forces, stress) -> Atoms:
    """Copy of `atoms` carrying a plain SinglePointCalculator (no live QE)."""
    from ase.calculators.singlepoint import SinglePointCalculator

    clean = atoms.copy()
    results = {"energy": float(energy), "forces": forces}
    if stress is not None:
        results["stress"] = stress
    clean.calc = SinglePointCalculator(clean, **results)
    return clean
