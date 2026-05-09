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
    → ROUND_CANDIDATES
    → ROUND_ACQUIRED
    → ROUND_DFT_QUEUED
    → ROUND_DFT_DONE
    → ROUND_TRAIN
    → ROUND_TRAINED
    → ROUND_VALIDATED
    → {loop r := r+1 OR}
    → FINAL_COMPILE
    → FINAL_VALIDATE
    → DONE

Plus terminal: PAUSED_PHYSICS (Tier-1 fail), CANCELLED (user request).

Every transition writes <run_dir>/.anvil/state.json atomically (write-temp +
fsync + rename) so a crash mid-write cannot corrupt the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


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


# Allowed transitions: from -> set of valid next states.
# Any "from" state can also transition to PAUSED_PHYSICS or CANCELLED (terminal
# fail-safe edges).
ALLOWED_TRANSITIONS: dict[AnvilState, set[AnvilState]] = {
    AnvilState.INIT: {AnvilState.BOOTSTRAP_SEEDS},
    AnvilState.BOOTSTRAP_SEEDS: {AnvilState.BOOTSTRAP_DFT_QUEUED},
    AnvilState.BOOTSTRAP_DFT_QUEUED: {AnvilState.BOOTSTRAP_DFT_DONE},
    AnvilState.BOOTSTRAP_DFT_DONE: {AnvilState.VAL_DFT_QUEUED},
    AnvilState.VAL_DFT_QUEUED: {AnvilState.VAL_DFT_DONE},
    AnvilState.VAL_DFT_DONE: {AnvilState.BOOTSTRAP_TRAIN},
    AnvilState.BOOTSTRAP_TRAIN: {AnvilState.BOOTSTRAP_TRAINED},
    AnvilState.BOOTSTRAP_TRAINED: {AnvilState.ROUND_CANDIDATES},
    AnvilState.ROUND_CANDIDATES: {AnvilState.ROUND_ACQUIRED},
    AnvilState.ROUND_ACQUIRED: {AnvilState.ROUND_DFT_QUEUED},
    AnvilState.ROUND_DFT_QUEUED: {AnvilState.ROUND_DFT_DONE},
    AnvilState.ROUND_DFT_DONE: {AnvilState.ROUND_TRAIN},
    AnvilState.ROUND_TRAIN: {AnvilState.ROUND_TRAINED},
    AnvilState.ROUND_TRAINED: {AnvilState.ROUND_VALIDATED},
    AnvilState.ROUND_VALIDATED: {
        AnvilState.ROUND_CANDIDATES,    # loop another round
        AnvilState.FINAL_COMPILE,        # done with AL
    },
    AnvilState.FINAL_COMPILE: {AnvilState.FINAL_VALIDATE},
    AnvilState.FINAL_VALIDATE: {AnvilState.DONE},
}

TERMINAL_STATES: set[AnvilState] = {
    AnvilState.DONE,
    AnvilState.PAUSED_PHYSICS,
    AnvilState.CANCELLED,
}


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

    # Content hashes for provenance (sha256 of canonical bytes)
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

    # Metadata
    schema_version: int = 1
    last_updated_iso: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict representation for JSON serialization."""
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StateCheckpoint":
        """Inverse of to_dict()."""
        d = dict(d)
        d["state"] = AnvilState(d["state"])
        d["monitor"] = [RoundMonitor(**m) for m in d.get("monitor", [])]
        # Drop unknown keys for forward-compat (newer schema reading older file).
        known = {f for f in cls.__dataclass_fields__}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)


class StateError(RuntimeError):
    """Raised on illegal state transition or corrupt checkpoint."""


def _checkpoint_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / ".anvil" / "state.json"


def save_checkpoint(run_dir: str | Path, ckpt: StateCheckpoint) -> Path:
    """Atomic write to <run_dir>/.anvil/state.json.

    Implementation: write to temp file in same dir, fsync, then os.replace().
    Ensures that a crash mid-write leaves either the old or new file intact —
    never a half-written one.
    """
    from datetime import datetime, timezone

    ckpt.last_updated_iso = datetime.now(timezone.utc).isoformat()
    path = _checkpoint_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(ckpt.to_dict(), indent=2, sort_keys=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".state.", suffix=".json.tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise

    return path


def load_checkpoint(run_dir: str | Path) -> StateCheckpoint:
    """Load from <run_dir>/.anvil/state.json. Raises FileNotFoundError if absent."""
    path = _checkpoint_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise StateError(f"Corrupt checkpoint at {path}: {exc}") from exc

    return StateCheckpoint.from_dict(data)


def transition(
    ckpt: StateCheckpoint,
    new_state: AnvilState,
    *,
    dft_calls_added: int = 0,
    monitor_entry: Optional[RoundMonitor] = None,
    advance_round: bool = False,
) -> StateCheckpoint:
    """Validate transition, return updated checkpoint (does NOT save).

    Caller is responsible for save_checkpoint(...) after the transition.

    Args:
        ckpt: current checkpoint
        new_state: target state
        dft_calls_added: increment dft_calls_used by this much
        monitor_entry: append a RoundMonitor record
        advance_round: increment round counter (caller passes True when entering
            the next round's CANDIDATES state)

    Raises StateError on illegal transitions.
    """
    if ckpt.state in TERMINAL_STATES:
        raise StateError(
            f"Cannot transition from terminal state {ckpt.state.value}"
        )

    # Terminal-edge fast path: any state can become PAUSED_PHYSICS or CANCELLED.
    if new_state not in TERMINAL_STATES:
        allowed = ALLOWED_TRANSITIONS.get(ckpt.state, set())
        if new_state not in allowed:
            raise StateError(
                f"Illegal transition: {ckpt.state.value} → {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

    ckpt.state = new_state
    ckpt.dft_calls_used += dft_calls_added
    if monitor_entry is not None:
        ckpt.monitor.append(monitor_entry)
    if advance_round:
        ckpt.round += 1

    return ckpt


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest. Used for provenance hashes (config, pools, models)."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    """SHA-256 of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
