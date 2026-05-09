"""State machine + JSON checkpointing.

See DESIGN.md §4.2.

The state machine:
    INIT
    → BOOTSTRAP_SEEDS
    → BOOTSTRAP_DFT_QUEUED
    → BOOTSTRAP_DFT_DONE
    → VAL_DFT_QUEUED
    → VAL_DFT_DONE
    → BOOTSTRAP_TRAIN
    → BOOTSTRAP_TRAINED
    → ROUND_r_CANDIDATES
    → ROUND_r_ACQUIRED
    → ROUND_r_DFT_QUEUED
    → ROUND_r_DFT_DONE
    → ROUND_r_TRAIN
    → ROUND_r_TRAINED
    → ROUND_r_VALIDATED
    → {loop r := r+1 OR}
    → FINAL_COMPILE
    → FINAL_VALIDATE
    → DONE

Plus terminal: PAUSED_PHYSICS (Tier-1 fail), CANCELLED (user request).

Every transition writes <run_dir>/.anvil/state.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class AnvilState(Enum):
    """Discrete states of an Anvil run. See DESIGN.md §4.2."""

    INIT = "init"
    BOOTSTRAP_SEEDS = "bootstrap_seeds"
    BOOTSTRAP_DFT_QUEUED = "bootstrap_dft_queued"
    BOOTSTRAP_DFT_DONE = "bootstrap_dft_done"
    VAL_DFT_QUEUED = "val_dft_queued"
    VAL_DFT_DONE = "val_dft_done"
    BOOTSTRAP_TRAIN = "bootstrap_train"
    BOOTSTRAP_TRAINED = "bootstrap_trained"
    ROUND_CANDIDATES = "round_candidates"
    ROUND_ACQUIRED = "round_acquired"
    ROUND_DFT_QUEUED = "round_dft_queued"
    ROUND_DFT_DONE = "round_dft_done"
    ROUND_TRAIN = "round_train"
    ROUND_TRAINED = "round_trained"
    ROUND_VALIDATED = "round_validated"
    FINAL_COMPILE = "final_compile"
    FINAL_VALIDATE = "final_validate"

    # Terminal
    DONE = "done"
    PAUSED_PHYSICS = "paused_physics"
    CANCELLED = "cancelled"


@dataclass
class RoundMonitor:
    """Per-round metrics tracked across the AL loop."""
    round: int
    pool_size: int
    dft_calls_used_cumulative: int
    val_e_mae: float = 0.0
    val_f_mae: float = 0.0
    val_s_mae: float = 0.0
    tier1_passed: bool = True
    tier1_failures: list[str] = field(default_factory=list)


@dataclass
class StateCheckpoint:
    """Full run state, serialized to .anvil/state.json on every transition."""

    run_id: str
    system: str
    state: AnvilState
    round: int = 0
    dft_calls_used: int = 0

    # Content hashes for provenance
    config_hash: str = ""
    pool_a_hash: str = ""
    pool_b_hash: str = ""
    pool_c_hash: str = ""
    val_set_hash: str = ""
    ensemble_hashes: list[str] = field(default_factory=list)

    # Slurm job tracking
    pending_job_ids: list[str] = field(default_factory=list)

    # Per-round history (for plateau detection + reporting)
    monitor: list[RoundMonitor] = field(default_factory=list)

    # Foundation + cluster
    foundation_path: str = ""
    cluster: str = ""


def save_checkpoint(run_dir: str | Path, ckpt: StateCheckpoint) -> None:
    """Serialize ckpt to <run_dir>/.anvil/state.json (atomic write)."""
    raise NotImplementedError("week 1: see DESIGN.md §4.2")


def load_checkpoint(run_dir: str | Path) -> StateCheckpoint:
    """Load from <run_dir>/.anvil/state.json. Raises FileNotFoundError if absent."""
    raise NotImplementedError("week 1: see DESIGN.md §4.2")


def transition(
    ckpt: StateCheckpoint,
    new_state: AnvilState,
    *,
    dft_calls_added: int = 0,
    monitor_entry: Optional[RoundMonitor] = None,
) -> StateCheckpoint:
    """Validate transition and return updated checkpoint.

    Raises StateError on illegal transitions.
    """
    raise NotImplementedError("week 1: see DESIGN.md §4.2")


class StateError(RuntimeError):
    """Raised on illegal state transition."""
