"""Orchestrator dry runs against the non-VASP engines.

Same path as test_orchestrator_dry_run.py, but with `dft.engine: qe` and
`dft.engine: ase` — proof that nothing downstream of `write_inputs` /
`collect` is VASP-specific.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

from anvil.state import AnvilState


@pytest.fixture(autouse=True)
def _stub_libfp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pools A/B are skipped here, but their modules still get imported."""
    if "libfp" not in sys.modules:
        try:
            import libfp  # noqa: F401
        except ImportError:
            monkeypatch.setitem(sys.modules, "libfp", types.ModuleType("libfp"))


def _seed(tmp_path: Path, name: str) -> Path:
    a = 3.5
    atoms = Atoms(
        symbols=["Si"] * 2,
        positions=[(0, 0, 0), (a / 2, a / 2, a / 2)],
        cell=[(a, 0, 0), (0, a, 0), (0, 0, a)],
        pbc=True,
    )
    p = tmp_path / f"{name}.vasp"
    ase_write(str(p), atoms, format="vasp", vasp5=True, direct=True)
    return p


def _bootstrap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dft_block: str,
               system: str):
    from anvil.orchestrator import Orchestrator

    seed_path = _seed(tmp_path, "si_dummy")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""\
system: {system}
elements: [Si]
seeds:
  from_local:
    - {seed_path}
{dft_block}
regime:
  pressures_gpa: [0, 10]
  temperatures_k: [300]
budget:
  max_dft_calls: 100
  max_walltime_h: 6
anchor:
  strain_factors: [0.97, 1.03]
"""
    )
    monkeypatch.chdir(tmp_path)
    orch = Orchestrator.from_config_path(
        cfg_path, cluster="amareln", scratch_root=tmp_path,
    )
    orch.bootstrap(do_submit=False, skip_pools=("A", "B"))
    assert orch.ckpt.state == AnvilState.BOOTSTRAP_DFT_QUEUED
    return orch


QE_DFT = """\
dft:
  engine: qe
  kspacing: 0.25
  engine_options:
    mode: pwx
    pseudo_dir: /pseudos
    pseudopotentials:
      Si: Si.pbe-n-kjpaw_psl.1.0.0.UPF
    slurm:
      ntasks: 8
"""

ASE_DFT = """\
dft:
  engine: ase
  engine_options:
    calculator: lj
    kwargs:
      sigma: 2.0
      epsilon: 0.1
      rc: 6.0
"""


def test_qe_engine_dry_run_writes_pw_inputs(tmp_path, monkeypatch):
    orch = _bootstrap(tmp_path, monkeypatch, QE_DFT, "si_qe")

    dirs = (orch.run_dir / "bootstrap_struct_dirs.txt").read_text().split()
    assert len(dirs) == 8                       # 4 anchor + 4 tier-2 validation
    for d in dirs:
        sd = Path(d)
        assert (sd / "pw.in").exists()
        assert not (sd / "INCAR").exists()       # nothing VASP-shaped
        meta = json.loads((sd / ".anvil_meta.json").read_text())
        assert meta["engine"] == "qe"
        assert meta["pool"] in ("C", "validation")
    assert "Si.pbe-n-kjpaw_psl.1.0.0.UPF" in (Path(dirs[0]) / "pw.in").read_text()


def test_ase_engine_full_bootstrap_cycle(tmp_path, monkeypatch):
    """write_inputs → run_local → collect, driven by the orchestrator."""
    orch = _bootstrap(tmp_path, monkeypatch, ASE_DFT, "si_ase")

    dirs = [Path(d) for d in
            (orch.run_dir / "bootstrap_struct_dirs.txt").read_text().split()]
    assert len(dirs) == 8

    engine = orch._make_engine()
    for sd in dirs:
        assert (sd / "input.xyz").exists()
        engine.run_local(sd)                     # stands in for the Slurm job

    orch.bootstrap_dft_done()
    assert orch.ckpt.state == AnvilState.BOOTSTRAP_DFT_DONE

    report = json.loads((orch.run_dir / "bootstrap_collect_report.json").read_text())
    assert report["train_collected"] == 4        # pool C
    assert report["val_collected"] == 4          # tier-2 validation
    assert report["train_failed"] == 0 and report["val_failed"] == 0

    train = ase_read(str(orch.run_dir / "train.xyz"), index=":", format="extxyz")
    assert len(train) == 4
    a = train[0]
    assert a.info["dft_engine"] == "ase"
    assert a.info["stress_convention"] == "ase"
    assert a.get_forces().shape == (len(a), 3)
    assert len(a.get_stress()) == 6
