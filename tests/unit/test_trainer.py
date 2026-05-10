"""Unit tests for anvil.ml.trainer — config materialization + Slurm script."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anvil.hpc.clusters.amareln import CLUSTER as AMARELN
from anvil.ml.trainer import (
    DEFAULT_SEEDS,
    EnsembleTrainResult,
    _materialize_cell_config,
    _slurm_train_script,
    _template_path,
    collect_compiled_models,
    train_ensemble,
)


def test_template_resolves() -> None:
    p = _template_path("allegro_oam_l_finetune")
    assert p.exists()
    assert p.suffix == ".yaml"


def test_template_has_required_keys() -> None:
    p = _template_path("allegro_oam_l_finetune")
    cfg = yaml.safe_load(p.read_text())
    assert "data" in cfg
    assert "trainer" in cfg
    # The 7 keys we mutate per cell
    assert "train_file_path" in cfg["data"]
    assert "val_file_path" in cfg["data"]
    assert "seed" in cfg["data"]
    assert "callbacks" in cfg["trainer"]
    assert "logger" in cfg["trainer"]


def test_materialize_mutates_correctly(tmp_path: Path) -> None:
    p = _template_path("allegro_oam_l_finetune")
    template_yaml = yaml.safe_load(p.read_text())
    train_xyz = tmp_path / "train.xyz"
    val_xyz = tmp_path / "val.xyz"
    train_xyz.write_text("dummy")
    val_xyz.write_text("dummy")

    run_dir = tmp_path / "ensemble_round0" / "member_s42"
    run_dir.mkdir(parents=True)

    cfg_path = _materialize_cell_config(
        template_yaml, train_xyz, val_xyz, seed=42,
        run_dir=run_dir, run_name="member_s42",
    )
    assert cfg_path.exists()
    out = yaml.safe_load(cfg_path.read_text())
    assert out["data"]["train_file_path"] == str(train_xyz.resolve())
    assert out["data"]["val_file_path"] == str(val_xyz.resolve())
    assert out["data"]["seed"] == 42
    assert out["trainer"]["callbacks"][0]["dirpath"].endswith(
        "/member_s42/checkpoints"
    )
    assert out["trainer"]["logger"]["name"] == "member_s42"


def test_slurm_script_has_required_directives(tmp_path: Path) -> None:
    run_dir = tmp_path / "member_s42"
    run_dir.mkdir()
    script, spec = _slurm_train_script(run_dir, AMARELN, time="06:00:00")
    body = script.read_text()
    assert "#SBATCH --gres=gpu:1" in body
    assert "#SBATCH --time=06:00:00" in body
    assert "nequip-train" in body
    assert "nequip-compile --mode torchscript --device cpu --target ase" in body
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in body
    assert "module load cuda/12.1.0" in body
    assert "conda activate nequip" in body


def test_train_ensemble_dry_run(tmp_path: Path) -> None:
    """do_submit=False writes run dirs + configs + sbp.sh without sbatch."""
    train_xyz = tmp_path / "train.xyz"
    val_xyz = tmp_path / "val.xyz"
    train_xyz.write_text("dummy")
    val_xyz.write_text("dummy")
    out_dir = tmp_path / "ensemble"

    result = train_ensemble(
        train_xyz=train_xyz, val_xyz=val_xyz,
        output_dir=out_dir, cluster=AMARELN,
        n_models=3, do_submit=False,
    )
    assert isinstance(result, EnsembleTrainResult)
    assert result.seeds == DEFAULT_SEEDS
    assert len(result.run_dirs) == 3
    assert len(result.job_ids) == 0
    for rd, seed in zip(result.run_dirs, DEFAULT_SEEDS):
        assert rd.name == f"member_s{seed}"
        assert (rd / "config.yaml").exists()
        assert (rd / "sbp.sh").exists()
        cfg = yaml.safe_load((rd / "config.yaml").read_text())
        assert cfg["data"]["seed"] == seed


def test_collect_compiled_models_skips_missing(tmp_path: Path) -> None:
    rd1 = tmp_path / "m1"
    rd2 = tmp_path / "m2"
    (rd1 / "checkpoints").mkdir(parents=True)
    (rd2 / "checkpoints").mkdir(parents=True)

    # Only rd1 has a compiled model
    p1 = rd1 / "checkpoints" / "best.nequip.pth"
    p1.write_bytes(b"junk")

    out = collect_compiled_models([rd1, rd2])
    assert out == [p1]


def test_n_models_seeds_mismatch(tmp_path: Path) -> None:
    train_xyz = tmp_path / "train.xyz"
    val_xyz = tmp_path / "val.xyz"
    train_xyz.write_text("dummy")
    val_xyz.write_text("dummy")
    with pytest.raises(ValueError, match="seeds length"):
        train_ensemble(
            train_xyz=train_xyz, val_xyz=val_xyz,
            output_dir=tmp_path / "e", cluster=AMARELN,
            n_models=3, seeds=[42, 123], do_submit=False,
        )
