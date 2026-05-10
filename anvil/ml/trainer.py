"""K-seed Allegro fine-tune trainer wrapping nequip's Hydra config.

See DESIGN.md §4.5, §1.1 #5 (reproducibility), §10 (operational lessons).

Each ensemble member shares the same train.xyz / val.xyz; only data.seed
differs. Lessons codified from the multi-seed paper experiment:
  - shadow-variable pattern around `conda activate` (avoids $SIZE clobber)
  - module load cuda/12.1.0 + intel/17.0.4 (provides libmpi for reformpy)
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (CUDA OOM mitigation)
  - nequip-compile --target ase (since nequip-deploy was removed in 0.16.x)

The template Hydra config is shipped under configs/training_templates/.
v1: configs/training_templates/allegro_oam_l_finetune.yaml — the Allegro-OAM-L
fine-tune recipe used in the prior paper, extracted verbatim and parameterized.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from anvil.hpc.clusters import ClusterConfig
from anvil.hpc.slurm import SlurmJobSpec, submit


DEFAULT_TEMPLATE = "allegro_oam_l_finetune"
DEFAULT_SEEDS: list[int] = [42, 123, 456]


def _template_path(name: str) -> Path:
    """Resolve configs/training_templates/<name>.yaml from package data."""
    here = Path(__file__).resolve().parent
    p = here.parent.parent / "configs" / "training_templates" / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"No training template {name!r} at {p}. "
            f"Available: {list((p.parent).glob('*.yaml'))}"
        )
    return p


def _materialize_cell_config(
    template_yaml: dict,
    train_xyz: str | Path,
    val_xyz: str | Path,
    seed: int,
    run_dir: Path,
    run_name: str,
) -> Path:
    """Clone template_yaml + mutate the per-cell paths and seed.

    Returns the path to the materialized config.yaml.
    """
    cfg = copy.deepcopy(template_yaml)
    cfg["data"]["train_file_path"] = str(Path(train_xyz).resolve())
    cfg["data"]["val_file_path"] = str(Path(val_xyz).resolve())
    cfg["data"]["seed"] = int(seed)

    ckpt_dir = run_dir / "checkpoints"
    log_root = run_dir.parent
    cfg["trainer"]["callbacks"][0]["dirpath"] = str(ckpt_dir.resolve())
    cfg["trainer"]["callbacks"][0]["filename"] = "best"
    cfg["trainer"]["logger"]["save_dir"] = str(log_root.resolve())
    cfg["trainer"]["logger"]["name"] = run_name

    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return cfg_path


def _slurm_train_script(
    run_dir: Path,
    cluster: ClusterConfig,
    *,
    time: str = "12:00:00",
    cuda_module: str = "cuda/12.1.0",
) -> tuple[Path, str]:
    """Build the per-cell training Slurm script.

    Codifies operational lessons:
      - module load cuda + intel (libmpi for reformpy if used downstream)
      - shadow-var pattern is unnecessary for trainer (we don't have $SIZE
        collisions), but PYTORCH_CUDA_ALLOC_CONF + PYTHONUNBUFFERED help.
      - nequip-train then nequip-compile (replaces nequip-deploy in 0.16.x).
    """
    spec = SlurmJobSpec(
        name=run_dir.name,
        partition=cluster.gpu_partition,
        n_tasks=1,
        cpus_per_task=4,
        mem="32G",
        time=time,
        output=str(run_dir / "train.out"),
        error=str(run_dir / "train.err"),
        gres="gpu:1",
        chdir=str(run_dir) if cluster.requires_chdir else "",
    )
    body = "\n".join([
        f"module load {cuda_module} 2>/dev/null || true",
        f"module load {cluster.intel_module} 2>/dev/null || true",
        f'eval "$({cluster.conda_root}/bin/conda shell.bash hook)"',
        f"conda activate {cluster.conda_env_train}",
        "export PYTHONUNBUFFERED=1",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f'cd "{run_dir}"',
        f'nequip-train --config-path="{run_dir}" --config-name=config',
        "",
        "# Compile best (or last) checkpoint to ASE-loadable .nequip.pth",
        f'CKPT_DIR="{run_dir}/checkpoints"',
        'if [ -f "$CKPT_DIR/best.ckpt" ]; then',
        '  nequip-compile --mode torchscript --device cpu --target ase '
            '"$CKPT_DIR/best.ckpt" "$CKPT_DIR/best.nequip.pth"',
        'elif [ -f "$CKPT_DIR/last.ckpt" ]; then',
        '  nequip-compile --mode torchscript --device cpu --target ase '
            '"$CKPT_DIR/last.ckpt" "$CKPT_DIR/last.nequip.pth"',
        'else',
        '  echo "ERROR: no checkpoint produced" >&2; exit 2',
        'fi',
    ])
    script = run_dir / "sbp.sh"
    script.write_text(spec.render(body))
    script.chmod(0o755)
    return script, spec


@dataclass
class EnsembleTrainResult:
    """What train_ensemble returns. Fields are persisted to checkpoint."""
    seeds: list[int] = field(default_factory=list)
    run_dirs: list[Path] = field(default_factory=list)
    job_ids: list[str] = field(default_factory=list)
    template: str = ""


def train_ensemble(
    train_xyz: str | Path,
    val_xyz: str | Path,
    output_dir: str | Path,
    *,
    cluster: ClusterConfig,
    n_models: int = 3,
    seeds: Optional[list[int]] = None,
    template: str = DEFAULT_TEMPLATE,
    time: str = "12:00:00",
    do_submit: bool = True,
    ssh_host: Optional[str] = None,
) -> EnsembleTrainResult:
    """Train K=n_models Allegro-OAM-L fine-tunes in parallel.

    Each member shares train_xyz / val_xyz; only data.seed differs. Returns
    an EnsembleTrainResult with run dirs + Slurm job IDs.

    The template config is shipped under configs/training_templates/. The
    foundation model path is INSIDE the template (set per cluster). This
    parameterization is intentionally minimal in v1; week-3 polish can move
    foundation_path out into a top-level field.
    """
    seeds = list(seeds) if seeds else DEFAULT_SEEDS[:n_models]
    if len(seeds) != n_models:
        raise ValueError(f"seeds length {len(seeds)} != n_models {n_models}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_path = _template_path(template)
    with open(template_path) as f:
        template_yaml = yaml.safe_load(f)

    result = EnsembleTrainResult(seeds=list(seeds), template=template)
    pending_scripts: list[Path] = []

    for seed in seeds:
        run_name = f"member_s{seed}"
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        result.run_dirs.append(run_dir)

        _materialize_cell_config(
            template_yaml, train_xyz, val_xyz, seed, run_dir, run_name,
        )
        script, _ = _slurm_train_script(run_dir, cluster, time=time)
        pending_scripts.append(script)

    if do_submit:
        for script in pending_scripts:
            jid = submit(script, ssh_host=ssh_host)
            result.job_ids.append(jid)

    return result


def collect_compiled_models(run_dirs: list[Path]) -> list[Path]:
    """For each run_dir, return path to checkpoints/best.nequip.pth (or
    last.nequip.pth) if present. Skips dirs where neither exists.
    """
    out = []
    for rd in run_dirs:
        ckpt = Path(rd) / "checkpoints"
        for cand in ("best.nequip.pth", "last.nequip.pth"):
            p = ckpt / cand
            if p.exists():
                out.append(p)
                break
    return out
