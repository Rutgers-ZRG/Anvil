"""VASP engine — wraps the v1 code path behind the DFTEngine interface.

Behaviour is unchanged from `anvil.dft.vasp` + `anvil.dft.runner`: INCAR from
`configs/functionals/<functional>.yaml`, gamma-centered KPOINTS from
`dft.kspacing`, concatenated POTCAR, `mpirun vasp_std` under Slurm, results
read back from vasprun.xml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ase import Atoms

from anvil.dft.engines.base import DFTEngine
from anvil.dft.vasp import load_functional, make_vasp_inputs, write_vasp_dir
from anvil.hpc.clusters import ClusterConfig


class VaspEngine(DFTEngine):
    """Plane-wave labeling with VASP (v1 default)."""

    name = "vasp"

    def __init__(
        self,
        *,
        cluster: Optional[ClusterConfig] = None,
        options: Optional[dict] = None,
        kspacing: float = 0.25,
        functional: str = "pbe",
        encut: int = 600,
        extra_incar: Optional[dict] = None,
        cat_potcar: bool = True,
        functional_yaml: Optional[dict] = None,
    ) -> None:
        super().__init__(cluster=cluster, options=options, kspacing=kspacing)
        self.functional = functional
        self.encut = encut
        self.extra_incar = extra_incar or None
        self.cat_potcar = cat_potcar
        self.functional_yaml = (
            functional_yaml if functional_yaml is not None else load_functional(functional)
        )
        # INCAR templates carry their own slurm block; user options win.
        merged_slurm = dict(self.functional_yaml.get("slurm") or {})
        merged_slurm.update(self.options.get("slurm") or {})
        self.options["slurm"] = merged_slurm

    def write_inputs(
        self,
        atoms: Atoms,
        out_dir: str | Path,
        *,
        sample_kind: str = "",
        pool: str = "",
    ) -> Path:
        potcar_root = self.options.get("potcar_root") or (
            self.cluster.potcar_root if self.cluster else "."
        )
        inp = make_vasp_inputs(
            atoms,
            functional=self.functional,
            encut=self.encut,
            kspacing=self.kspacing,
            extra_incar=self.extra_incar,
            potcar_root=potcar_root,
            sample_kind=sample_kind,
            pool=pool,
        )
        out = write_vasp_dir(inp, out_dir, cat_potcar=self.cat_potcar)
        # write_vasp_dir already wrote .anvil_meta.json with the POTCAR paths;
        # re-write it through the base class so it also carries `engine`.
        self.write_meta(
            out, sample_kind=sample_kind, pool=pool,
            extra={"potcar_paths": inp.potcar_paths},
        )
        return out

    def _job_body(self, struct_dir: Path) -> str:
        vasp_bin = self.options.get("vasp_bin") or (
            self.cluster.vasp_bin if self.cluster else "vasp_std"
        )
        spec = self._job_spec(struct_dir)
        lines = [
            *self._module_lines(),
            *self._env_lines(),
            f"cd {struct_dir}",
            f"{self._launcher(spec.n_tasks)} {vasp_bin} > vasp.log 2>&1",
        ]
        return "\n".join(lines)

    def is_converged(self, struct_dir: str | Path) -> bool:
        """True if OUTCAR reports a completed run."""
        outcar = Path(struct_dir) / "OUTCAR"
        if not outcar.exists():
            return False
        try:
            text = outcar.read_text(errors="ignore")
        except OSError:
            return False
        return ("Total CPU time" in text) or ("reached required accuracy" in text)

    def _read_result(self, struct_dir: Path) -> Atoms:
        from ase.io import read as ase_read

        vp = struct_dir / "vasprun.xml"
        if not vp.exists():
            raise FileNotFoundError(f"No vasprun.xml in {struct_dir}")
        # vasprun.xml gives ASE-convention (compression-negative) stress.
        return ase_read(str(vp), format="vasp-xml")
