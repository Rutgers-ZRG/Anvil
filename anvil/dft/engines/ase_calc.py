"""Generic ASE-calculator engine — label with *any* ASE calculator.

This is the escape hatch that makes Anvil code-agnostic: if a code has an ASE
calculator, Anvil can use it as the labeling backend without a bespoke engine.
CP2K, GPAW, Abinit, FHI-aims, Psi4, ORCA, xTB, or a cheap analytic potential
(EMT / Lennard-Jones) for pipeline smoke tests all work.

Config sketch::

    dft:
      engine: ase
      engine_options:
        calculator: gpaw.GPAW              # "module.path.ClassName", or a
                                           # shortcut: emt / lj / espresso / cp2k
        kwargs:
          mode: {name: pw, ecut: 600}
          xc: PBE
        use_kpts: true                     # pass kpts= from dft.kspacing
        slurm: {ntasks: 32, mem: 64G, time: "03:00:00"}

Calculators that need Python objects rather than plain values take an
`__object__` key naming the class to build::

    calculator: ase.calculators.espresso.Espresso
    kwargs:
      profile:
        __object__: ase.calculators.espresso.EspressoProfile
        command: "srun --mpi=pmi2 /path/to/pw.x"
        pseudo_dir: /path/to/pseudos

The structure is written as `input.xyz`; the calculator spec is recorded in
`.anvil_meta.json` so `python -m anvil.dft.engines._run <struct_dir>` can
rebuild the calculator on a compute node. Results land in `anvil_result.xyz`
in ASE units and ASE stress convention, like every other engine.
"""

from __future__ import annotations

import importlib
import shlex
from pathlib import Path
from typing import Optional

from ase import Atoms

from anvil.dft.engines.base import RESULT_FILE, DFTEngine, EngineError
from anvil.dft.vasp import kpoint_mesh
from anvil.hpc.clusters import ClusterConfig

INPUT_XYZ = "input.xyz"

# Convenience names so common backends need no import path in the yaml.
CALCULATOR_SHORTCUTS: dict[str, str] = {
    "emt": "ase.calculators.emt.EMT",
    "lj": "ase.calculators.lj.LennardJones",
    "espresso": "ase.calculators.espresso.Espresso",
    "cp2k": "ase.calculators.cp2k.CP2K",
    "gpaw": "gpaw.GPAW",
    "abinit": "ase.calculators.abinit.Abinit",
    "aims": "ase.calculators.aims.Aims",
    "vasp": "ase.calculators.vasp.Vasp",
}


OBJECT_KEY = "__object__"


def build_kwargs(spec):
    """Recursively turn yaml into calculator kwargs.

    Any mapping carrying an `__object__` key is instantiated::

        profile:
          __object__: ase.calculators.espresso.EspressoProfile
          command: "srun --mpi=pmi2 pw.x"
          pseudo_dir: /path/to/pseudos

    which is what calculators like Espresso (profile=) and GPAW (mode=) need
    and plain yaml cannot otherwise express.
    """
    if isinstance(spec, dict):
        if OBJECT_KEY in spec:
            cls = resolve_calculator_class(spec[OBJECT_KEY])
            rest = {k: build_kwargs(v) for k, v in spec.items() if k != OBJECT_KEY}
            return cls(**rest)
        return {k: build_kwargs(v) for k, v in spec.items()}
    if isinstance(spec, (list, tuple)):
        return type(spec)(build_kwargs(v) for v in spec)
    return spec


def resolve_calculator_class(spec: str):
    """Import a calculator class from "module.path.ClassName" or a shortcut."""
    path = CALCULATOR_SHORTCUTS.get(str(spec).lower(), str(spec))
    if ":" in path:
        module_name, cls_name = path.split(":", 1)
    elif "." in path:
        module_name, cls_name = path.rsplit(".", 1)
    else:
        raise EngineError(
            f"Cannot resolve calculator {spec!r}: give a full import path like "
            f"'gpaw.GPAW', or one of {sorted(CALCULATOR_SHORTCUTS)}."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EngineError(f"Cannot import {module_name!r} for calculator {spec!r}") from exc
    try:
        return getattr(module, cls_name)
    except AttributeError as exc:
        raise EngineError(f"{module_name!r} has no attribute {cls_name!r}") from exc


class ASECalculatorEngine(DFTEngine):
    """Label structures with an arbitrary ASE calculator."""

    name = "ase"

    def __init__(
        self,
        *,
        cluster: Optional[ClusterConfig] = None,
        options: Optional[dict] = None,
        kspacing: float = 0.25,
    ) -> None:
        super().__init__(cluster=cluster, options=options, kspacing=kspacing)
        self.calculator_spec = self.options.get("calculator")
        if not self.calculator_spec:
            raise EngineError(
                "engine 'ase' needs dft.engine_options.calculator "
                "(e.g. 'gpaw.GPAW', 'ase.calculators.cp2k.CP2K', or 'emt')."
            )
        self.calc_kwargs: dict = dict(self.options.get("kwargs") or {})
        self.use_kpts = bool(self.options.get("use_kpts", False))

    # ---------------------------------------------------------------- inputs

    def write_inputs(
        self,
        atoms: Atoms,
        out_dir: str | Path,
        *,
        sample_kind: str = "",
        pool: str = "",
    ) -> Path:
        from ase.io import write as ase_write

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ase_write(str(out / INPUT_XYZ), atoms, format="extxyz")
        kwargs = dict(self.calc_kwargs)
        if self.use_kpts:
            kwargs["kpts"] = kpoint_mesh(atoms, self.kspacing)
        self.write_meta(
            out, sample_kind=sample_kind, pool=pool,
            extra={"calculator": self.calculator_spec, "kwargs": kwargs},
        )
        return out

    # ------------------------------------------------------------ submission

    def _job_body(self, struct_dir: Path) -> str:
        spec = self._job_spec(struct_dir)
        python_bin = self._python_bin()
        runner = (
            f"{python_bin} -m anvil.dft.engines._run {shlex.quote(str(struct_dir))}"
        )
        # Most ASE calculators drive their own MPI binary; only launch the
        # Python process itself under MPI when the user asks for it (GPAW).
        if self.options.get("mpi_python", False):
            runner = f"{self._launcher(spec.n_tasks)} {runner}"
        return "\n".join([
            *self._module_lines(),
            *self._env_lines(),
            *self._conda_lines(),
            f"cd {struct_dir}",
            f"{runner} > ase_calc.log 2>&1",
        ])

    # -------------------------------------------------------------- labeling

    def build_calculator(self, struct_dir: str | Path):
        """Instantiate the configured ASE calculator for one struct dir."""
        meta = self.read_meta(struct_dir)
        spec = meta.get("calculator") or self.calculator_spec
        kwargs = meta.get("kwargs")
        if kwargs is None:
            kwargs = dict(self.calc_kwargs)
        cls = resolve_calculator_class(spec)
        return cls(**build_kwargs(kwargs))

    def run_local(self, struct_dir: str | Path) -> Atoms:
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.io import read as ase_read

        sd = Path(struct_dir)
        xyz = sd / INPUT_XYZ
        if not xyz.exists():
            raise FileNotFoundError(f"No {INPUT_XYZ} in {sd} — call write_inputs first")
        atoms = ase_read(str(xyz), format="extxyz")
        atoms.calc = self.build_calculator(sd)
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        try:
            stress = atoms.get_stress()
        except Exception:
            stress = None  # e.g. molecular calculators with no cell

        labeled = atoms.copy()
        results = {"energy": energy, "forces": forces}
        if stress is not None:
            results["stress"] = stress
        labeled.calc = SinglePointCalculator(labeled, **results)
        self.write_result(sd, labeled)
        return labeled

    # ------------------------------------------------------------ collection

    def is_converged(self, struct_dir: str | Path) -> bool:
        return (Path(struct_dir) / RESULT_FILE).exists()

    def _read_result(self, struct_dir: Path) -> Atoms:
        from ase.io import read as ase_read

        result = struct_dir / RESULT_FILE
        if not result.exists():
            raise FileNotFoundError(f"No {RESULT_FILE} in {struct_dir}")
        return ase_read(str(result), format="extxyz")
