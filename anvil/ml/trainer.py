"""K-seed fine-tune trainer wrapping nequip's Hydra config.

See DESIGN.md §4.5, §1.1 #5 (reproducibility).

Each ensemble member trained with same hyperparameters, different `data.seed`.
Warm-start from previous round's checkpoints by default; full restart every
3 rounds for safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def train_ensemble(
    pool_xyz: str | Path,
    val_xyz: str | Path,
    n_models: int,
    seeds: list[int],
    output_dir: str | Path,
    *,
    foundation_path: str,
    loss_weights: str = "1:1:0.01",
    learning_rate: float = 5e-5,
    patience: int = 30,
    max_epochs: int = 200,
    warm_start_dir: Optional[str | Path] = None,
) -> list[Path]:
    """Train K=n_models members, return list of best.nequip.pth paths.

    Lessons codified:
    - conda env activation uses shadow-variable pattern (anvil.hpc.env).
    - $SIZE clobber avoided via __MS_* shadow before activate.
    - nequip-compile --target ase (since nequip-deploy was removed in 0.16.x).
    """
    raise NotImplementedError("week 1: see DESIGN.md §4.5")
