"""End-to-end dry-run: orchestrator INIT → BOOTSTRAP_DFT_QUEUED with no_submit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write as ase_write

from anvil.orchestrator import Orchestrator
from anvil.state import AnvilState


def _seed(tmp_path: Path, name: str) -> Path:
    a = 3.5
    atoms = Atoms(
        symbols=["Si"] * 2,
        positions=[(0, 0, 0), (a/2, a/2, a/2)],
        cell=[(a, 0, 0), (0, a, 0), (0, 0, a)],
        pbc=True,
    )
    p = tmp_path / f"{name}.vasp"
    ase_write(str(p), atoms, format="vasp", vasp5=True, direct=True)
    return p


def test_dry_run_si_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a minimal yaml with local Si seed
    seed_path = _seed(tmp_path, "si_dummy")
    cfg_path = tmp_path / "si.yaml"
    cfg_path.write_text(
        f"""\
system: si
elements: [Si]
seeds:
  from_local:
    - {seed_path}
dft:
  functional: pbe
  encut: 600
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

    # Use tmp_path as scratch_root, run from tmp_path so .anvil/ goes there
    monkeypatch.chdir(tmp_path)

    orch = Orchestrator.from_config_path(
        cfg_path, cluster="amareln", scratch_root=tmp_path,
    )
    assert orch.ckpt.state == AnvilState.INIT

    # Bootstrap WITHOUT submitting and WITHOUT foundation MLIP (no model
    # available in unit-test env). Pool C and validation are foundation-free.
    orch.bootstrap(do_submit=False, skip_pools=("A", "B"))
    assert orch.ckpt.state == AnvilState.BOOTSTRAP_DFT_QUEUED

    # Pool C: 1 phase × 2 P × 2 strain = 4 cells
    # Tier-2: 1 phase × 2 P × 2 (default 0.92, 1.08) = 4 cells
    # Total = 8 struct dirs
    manifest = (orch.run_dir / "bootstrap_struct_dirs.txt").read_text().splitlines()
    assert len(manifest) == 8
    for d in manifest:
        sd = Path(d)
        assert (sd / "INCAR").exists()
        assert (sd / "KPOINTS").exists()
        assert (sd / "POSCAR").exists()
        meta = json.loads((sd / ".anvil_meta.json").read_text())
        assert meta["pool"] in ("C", "validation")

    # bootstrap_manifest.json should be present and group dirs by pool
    pool_manifest = json.loads((orch.run_dir / "bootstrap_manifest.json").read_text())
    assert len(pool_manifest["pool_a"]) == 0       # skipped
    assert len(pool_manifest["pool_b"]) == 0       # skipped
    assert len(pool_manifest["pool_c"]) == 4
    assert len(pool_manifest["validation"]) == 4

    # Checkpoint persisted, .anvil symlink in cwd
    state = json.loads((orch.run_dir / ".anvil" / "state.json").read_text())
    assert state["state"] == "bootstrap_dft_queued"
    assert state["dft_calls_used"] == 8
    assert (tmp_path / ".anvil" / "si").is_symlink()
