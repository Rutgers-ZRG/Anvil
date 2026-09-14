"""Engine-abstraction tests: registry, QE input writing, ASE-calculator runs."""

from __future__ import annotations

import json

import pytest
import yaml
from ase.build import bulk

from anvil.config import ConfigError, DFTConfig, load_config
from anvil.dft.engines import (
    EngineError,
    engine_from_struct_dir,
    get_engine_class,
    make_engine,
)
from anvil.dft.engines.ase_calc import ASECalculatorEngine, resolve_calculator_class
from anvil.dft.engines.base import META_FILE, RESULT_FILE
from anvil.dft.engines.qe import PW_INPUT, QEEngine
from anvil.dft.engines.vasp_engine import VaspEngine


# --------------------------------------------------------------- registry


def test_registry_resolves_all_engines():
    assert get_engine_class("vasp") is VaspEngine
    assert get_engine_class("qe") is QEEngine
    assert get_engine_class("qepy") is QEEngine       # alias
    assert get_engine_class("ase") is ASECalculatorEngine


def test_unknown_engine_raises():
    with pytest.raises(EngineError, match="Unknown DFT engine"):
        get_engine_class("siesta_maybe")


def test_make_engine_defaults_to_vasp():
    engine = make_engine(DFTConfig())
    assert isinstance(engine, VaspEngine)
    assert engine.name == "vasp"


# ------------------------------------------------------------------- QE


def _qe_engine(**opts) -> QEEngine:
    options = {
        "pseudo_dir": "/pseudos",
        "pseudopotentials": {"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"},
        "input_data": {"system": {"ecutwfc": 42.0}},
    }
    options.update(opts)
    return QEEngine(options=options, kspacing=0.25)


def test_qe_write_inputs_emits_pw_in(tmp_path):
    engine = _qe_engine()
    atoms = bulk("Si", "diamond", a=5.43)
    out = engine.write_inputs(atoms, tmp_path / "struct_0000", sample_kind="strain_0.95",
                              pool="C")

    text = (out / PW_INPUT).read_text()
    assert "calculation" in text and "'scf'" in text
    assert "ecutwfc" in text and "42" in text          # user override applied
    assert "ecutrho" in text                            # default kept
    assert "Si.pbe-n-kjpaw_psl.1.0.0.UPF" in text
    assert "K_POINTS automatic" in text
    assert "tstress" in text and "tprnfor" in text

    meta = json.loads((out / META_FILE).read_text())
    assert meta["engine"] == "qe"
    assert meta["sample_kind"] == "strain_0.95"
    assert meta["pool"] == "C"
    assert len(meta["kpts"]) == 3


def test_qe_missing_pseudopotential_is_an_error(tmp_path):
    engine = _qe_engine()
    with pytest.raises(EngineError, match="No pseudopotential"):
        engine.write_inputs(bulk("Cu", "fcc", a=3.6), tmp_path / "s")


def test_qe_bad_mode_rejected():
    with pytest.raises(EngineError, match="must be 'qepy' or 'pwx'"):
        _qe_engine(mode="carrier_pigeon")


def test_qe_job_body_qepy_mode_runs_the_python_runner(tmp_path):
    engine = _qe_engine(mode="qepy", slurm={"ntasks": 8})
    sd = tmp_path / "struct_0000"
    engine.write_inputs(bulk("Si", "diamond", a=5.43), sd)
    script, spec = engine.make_job_script(sd)
    body = script.read_text()
    assert "mpirun -n 8" in body
    assert "anvil.dft.engines._run" in body
    assert spec.n_tasks == 8


def test_qe_job_body_pwx_mode_runs_pw_x(tmp_path):
    engine = _qe_engine(mode="pwx", pw_bin="/apps/qe/bin/pw.x", slurm={"ntasks": 16})
    sd = tmp_path / "struct_0000"
    engine.write_inputs(bulk("Si", "diamond", a=5.43), sd)
    body = (engine.make_job_script(sd)[0]).read_text()
    assert "/apps/qe/bin/pw.x -in pw.in > pw.out" in body
    assert "anvil.dft.engines._run" not in body


def test_qe_convergence_from_pw_out(tmp_path):
    engine = _qe_engine(mode="pwx")
    sd = tmp_path / "struct_0000"
    sd.mkdir()
    assert not engine.is_converged(sd)

    (sd / "pw.out").write_text("!    total energy = -1.0 Ry\nJOB DONE.\n")
    assert engine.is_converged(sd)

    (sd / "pw.out").write_text(
        "convergence NOT achieved after 200 iterations\nJOB DONE.\n"
    )
    assert not engine.is_converged(sd)


# ---------------------------------------------------- generic ASE calculator


def test_resolve_calculator_shortcuts_and_paths():
    from ase.calculators.emt import EMT

    assert resolve_calculator_class("emt") is EMT
    assert resolve_calculator_class("ase.calculators.emt.EMT") is EMT
    assert resolve_calculator_class("ase.calculators.emt:EMT") is EMT
    with pytest.raises(EngineError):
        resolve_calculator_class("nonexistent_module.Calc")


def test_ase_engine_requires_a_calculator():
    with pytest.raises(EngineError, match="needs dft.engine_options.calculator"):
        ASECalculatorEngine(options={})


def test_ase_engine_labels_and_collects(tmp_path):
    """Full write → run → collect cycle with EMT standing in for a DFT code."""
    engine = ASECalculatorEngine(options={"calculator": "emt"})
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    sd = tmp_path / "struct_0000"
    engine.write_inputs(atoms, sd, sample_kind="anchor_1.00", pool="C")

    assert not engine.is_converged(sd)
    engine.run_local(sd)
    assert engine.is_converged(sd)
    assert (sd / RESULT_FILE).exists()

    collected, failed = engine.collect([sd])
    assert failed == []
    assert len(collected) == 1
    a = collected[0]
    assert a.get_forces().shape == (len(atoms), 3)
    assert len(a.get_stress()) == 6
    assert a.info["stress_convention"] == "ase"
    assert a.info["dft_engine"] == "ase"
    assert a.info["sample_kind"] == "anchor_1.00"
    assert a.info["pool"] == "C"


def test_collect_drops_unconverged_and_insane_forces(tmp_path):
    from ase.calculators.singlepoint import SinglePointCalculator
    import numpy as np

    engine = ASECalculatorEngine(options={"calculator": "emt"})
    good = tmp_path / "good"
    engine.write_inputs(bulk("Cu", "fcc", a=3.6, cubic=True), good)
    engine.run_local(good)

    unrun = tmp_path / "unrun"
    engine.write_inputs(bulk("Cu", "fcc", a=3.6, cubic=True), unrun)

    exploded = tmp_path / "exploded"
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    engine.write_inputs(atoms, exploded)
    blown = atoms.copy()
    forces = np.zeros((len(blown), 3))
    forces[0, 0] = 5.0e3
    blown.calc = SinglePointCalculator(blown, energy=-1.0, forces=forces)
    engine.write_result(exploded, blown)

    collected, failed = engine.collect([good, unrun, exploded])
    assert len(collected) == 1
    assert set(failed) == {str(unrun), str(exploded)}


def test_engine_from_struct_dir_round_trip(tmp_path):
    """A compute node can rebuild the engine from .anvil_meta.json alone."""
    engine = ASECalculatorEngine(options={"calculator": "emt", "kwargs": {}})
    sd = tmp_path / "struct_0000"
    engine.write_inputs(bulk("Cu", "fcc", a=3.6, cubic=True), sd)

    rebuilt = engine_from_struct_dir(sd)
    assert isinstance(rebuilt, ASECalculatorEngine)
    rebuilt.run_local(sd)
    assert rebuilt.is_converged(sd)


def test_run_module_labels_a_struct_dir(tmp_path):
    from anvil.dft.engines import _run

    engine = ASECalculatorEngine(options={"calculator": "emt"})
    sd = tmp_path / "struct_0000"
    engine.write_inputs(bulk("Cu", "fcc", a=3.6, cubic=True), sd)

    assert _run.main([str(sd)]) == 0
    assert (sd / RESULT_FILE).exists()


# ------------------------------------------------------------ config wiring


def _base_cfg(dft: dict) -> dict:
    return {
        "system": "si",
        "elements": ["Si"],
        "seeds": {"from_local": ["si.vasp"]},
        "dft": dft,
        "regime": {"pressures_gpa": [0], "temperatures_k": [300]},
        "budget": {"max_dft_calls": 10, "max_walltime_h": 1},
    }


def _write_cfg(tmp_path, dft: dict):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_base_cfg(dft)))
    return path


def test_config_default_engine_is_vasp(tmp_path):
    cfg = load_config(_write_cfg(tmp_path, {"functional": "pbe"}))
    assert cfg.dft.engine == "vasp"


def test_config_accepts_qe_engine(tmp_path):
    cfg = load_config(_write_cfg(tmp_path, {
        "engine": "qe",
        "engine_options": {
            "pseudo_dir": "/pseudos",
            "pseudopotentials": {"Si": "Si.upf"},
        },
    }))
    assert cfg.dft.engine == "qe"
    assert isinstance(make_engine(cfg.dft), QEEngine)


def test_config_rejects_unknown_engine(tmp_path):
    with pytest.raises(ConfigError, match="Unknown `dft.engine`"):
        load_config(_write_cfg(tmp_path, {"engine": "abinitio_by_hand"}))


def test_config_qe_requires_pseudopotentials(tmp_path):
    with pytest.raises(ConfigError, match="pseudopotentials"):
        load_config(_write_cfg(tmp_path, {
            "engine": "qe", "engine_options": {"pseudo_dir": "/pseudos"},
        }))


def test_config_ase_requires_calculator(tmp_path):
    with pytest.raises(ConfigError, match="calculator"):
        load_config(_write_cfg(tmp_path, {"engine": "ase", "engine_options": {}}))


def test_shipped_example_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "configs" / "examples"
    for name in ("si_qe.yaml", "si_ase.yaml"):
        cfg = load_config(root / name)
        assert cfg.dft.engine in ("qe", "ase")


# ------------------------------------------------------------------ VASP


def _dummy_cluster():
    from anvil.hpc.clusters import ClusterConfig

    return ClusterConfig(name="test", hostname="localhost", vasp_bin="/apps/vasp_std")


def test_vasp_engine_writes_inputs_and_script(tmp_path):
    engine = VaspEngine(
        cluster=_dummy_cluster(),
        functional="pbe",
        encut=520,
        cat_potcar=False,
        kspacing=0.25,
    )
    sd = tmp_path / "struct_0000"
    engine.write_inputs(bulk("Si", "diamond", a=5.43), sd, sample_kind="md", pool="A")

    assert "ENCUT = 520" in (sd / "INCAR").read_text()
    assert (sd / "KPOINTS").exists() and (sd / "POSCAR").exists()
    assert not (sd / "POTCAR").exists()               # cat_potcar=False
    assert json.loads((sd / META_FILE).read_text())["engine"] == "vasp"

    body = engine.make_job_script(sd)[0].read_text()
    assert "/apps/vasp_std" in body
    assert "module load intel/17.0.4" in body         # from configs/functionals/pbe.yaml
    assert "#SBATCH --ntasks=32" in body


def test_vasp_engine_is_converged(tmp_path):
    engine = VaspEngine(cluster=_dummy_cluster())
    sd = tmp_path / "s"
    sd.mkdir()
    assert not engine.is_converged(sd)
    (sd / "OUTCAR").write_text("...\n Total CPU time used (sec): 12.3\n")
    assert engine.is_converged(sd)


def test_runner_wrappers_still_work(tmp_path):
    """The pre-engine helpers keep their old behaviour."""
    from anvil.dft.runner import is_converged, make_vasp_slurm_script

    sd = tmp_path / "struct_0000"
    sd.mkdir()
    script, spec = make_vasp_slurm_script(
        sd, _dummy_cluster(), {"slurm": {"ntasks": 8, "partition": "main"}},
        job_name="job42",
    )
    body = script.read_text()
    assert "mpirun -n 8 /apps/vasp_std > vasp.log" in body
    assert "#SBATCH --job-name=job42" in body
    assert spec.n_tasks == 8
    assert not is_converged(sd)


def test_runner_uses_a_real_interpreter_path(tmp_path):
    """Job scripts must not assume a bare `python` exists on the node."""
    import sys

    engine = _qe_engine(mode="qepy")
    sd = tmp_path / "struct_0000"
    engine.write_inputs(bulk("Si", "diamond", a=5.43), sd)
    assert sys.executable in engine.make_job_script(sd)[0].read_text()

    override = _qe_engine(mode="qepy", python_bin="/opt/conda/envs/qepy/bin/python")
    override.write_inputs(bulk("Si", "diamond", a=5.43), sd)
    assert "/opt/conda/envs/qepy/bin/python -m anvil.dft.engines._run" in (
        override.make_job_script(sd)[0].read_text()
    )


# ------------------------------------------- cluster-specific job rendering


def test_amareln_qe_job_matches_the_cluster_conventions(tmp_path):
    """QE on amareln: srun launcher, its own intel module, group QE 7.2 build.

    VASP on the same cluster keeps mpirun and intel/17.0.4 — the two engines
    must not share a launcher.
    """
    from anvil.config import load_config
    from anvil.hpc.clusters import get_cluster
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "examples" / "si_qe_amareln.yaml")
    cluster = get_cluster("amareln")
    engine = make_engine(cfg.dft, cluster=cluster)

    sd = tmp_path / "struct_0000"
    engine.write_inputs(bulk("Si", "diamond", a=5.43), sd)
    body = engine.make_job_script(sd)[0].read_text()

    assert "srun --mpi=pmi2 /home/lz432/apps/q-e-qe-7.2/bin/pw.x -pd .true. -in pw.in" in body
    assert "module purge" in body
    assert "module load intel/17.0.2" in body       # QE build, not VASP's 17.0.4
    assert "ulimit -s unlimited" in body
    assert "export OMP_NUM_THREADS=1" in body
    assert "#SBATCH --mem-per-cpu=4GB" in body
    assert "mpirun" not in body

    # pseudo_dir comes from the cluster registry, not the yaml
    assert "'/home/mw1134/projects/qe_potential'" in (sd / "pw.in").read_text()

    # ...while VASP on the same cluster is untouched
    vasp_engine = VaspEngine(cluster=cluster, cat_potcar=False)
    vsd = tmp_path / "vasp_0000"
    vasp_engine.write_inputs(bulk("Si", "diamond", a=5.43), vsd)
    vbody = vasp_engine.make_job_script(vsd)[0].read_text()
    assert "mpirun -n 32 /home/lz432/apps/vasp.6.4.2/bin/vasp_std" in vbody
    assert "module load intel/17.0.4" in vbody
    assert "srun" not in vbody


def test_qe_without_any_pseudo_dir_fails_loudly(tmp_path):
    engine = QEEngine(options={"pseudopotentials": {"Si": "si.UPF"}})
    with pytest.raises(EngineError, match="No pseudopotential directory"):
        engine.write_inputs(bulk("Si", "diamond", a=5.43), tmp_path / "s")
