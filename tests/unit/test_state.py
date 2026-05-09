"""Unit tests for anvil.state — JSON round-trip + transition validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from anvil.state import (
    AnvilState,
    RoundMonitor,
    StateCheckpoint,
    StateError,
    hash_bytes,
    hash_file,
    load_checkpoint,
    save_checkpoint,
    transition,
)


def make_ckpt() -> StateCheckpoint:
    return StateCheckpoint(
        run_id="20260509-123456-carbon-abcd",
        system="carbon",
        state=AnvilState.INIT,
        config_hash=hash_bytes(b"sample"),
        cluster="amareln",
    )


def test_to_dict_round_trip() -> None:
    ckpt = make_ckpt()
    ckpt.monitor.append(RoundMonitor(round=1, pool_size=180, dft_calls_used_cumulative=180))
    d = ckpt.to_dict()
    assert d["state"] == "init"
    assert d["system"] == "carbon"
    assert len(d["monitor"]) == 1
    rt = StateCheckpoint.from_dict(d)
    assert rt.state == AnvilState.INIT
    assert rt.monitor[0].pool_size == 180


def test_save_load_atomic(tmp_path: Path) -> None:
    ckpt = make_ckpt()
    ckpt.dft_calls_used = 42
    save_checkpoint(tmp_path, ckpt)

    json_path = tmp_path / ".anvil" / "state.json"
    assert json_path.exists()
    # No leftover .tmp files from atomic-write
    siblings = list(json_path.parent.iterdir())
    assert all(p.name == "state.json" for p in siblings), f"Stray temp files: {siblings}"

    rt = load_checkpoint(tmp_path)
    assert rt.run_id == ckpt.run_id
    assert rt.dft_calls_used == 42
    assert rt.last_updated_iso != ""


def test_load_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path)


def test_load_corrupt(tmp_path: Path) -> None:
    bad = tmp_path / ".anvil" / "state.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not valid json")
    with pytest.raises(StateError):
        load_checkpoint(tmp_path)


def test_transition_valid_chain() -> None:
    ckpt = make_ckpt()
    transition(ckpt, AnvilState.BOOTSTRAP_SEEDS)
    assert ckpt.state == AnvilState.BOOTSTRAP_SEEDS
    transition(ckpt, AnvilState.BOOTSTRAP_DFT_QUEUED, dft_calls_added=180)
    assert ckpt.state == AnvilState.BOOTSTRAP_DFT_QUEUED
    assert ckpt.dft_calls_used == 180


def test_transition_illegal() -> None:
    ckpt = make_ckpt()
    with pytest.raises(StateError):
        transition(ckpt, AnvilState.BOOTSTRAP_TRAIN)  # skipping states


def test_transition_to_terminal_anytime() -> None:
    ckpt = make_ckpt()
    transition(ckpt, AnvilState.BOOTSTRAP_SEEDS)
    # Terminal fail-safe edges allowed from any non-terminal state
    transition(ckpt, AnvilState.PAUSED_PHYSICS)
    assert ckpt.state == AnvilState.PAUSED_PHYSICS


def test_no_transitions_from_terminal() -> None:
    ckpt = make_ckpt()
    transition(ckpt, AnvilState.BOOTSTRAP_SEEDS)
    transition(ckpt, AnvilState.CANCELLED)
    with pytest.raises(StateError):
        transition(ckpt, AnvilState.BOOTSTRAP_DFT_QUEUED)


def test_round_loop_back() -> None:
    ckpt = make_ckpt()
    bootstrap = [
        AnvilState.BOOTSTRAP_SEEDS,
        AnvilState.BOOTSTRAP_DFT_QUEUED,
        AnvilState.BOOTSTRAP_DFT_DONE,
        AnvilState.VAL_DFT_QUEUED,
        AnvilState.VAL_DFT_DONE,
        AnvilState.BOOTSTRAP_TRAIN,
        AnvilState.BOOTSTRAP_TRAINED,
    ]
    round_chain = [
        AnvilState.ROUND_CANDIDATES,
        AnvilState.ROUND_ACQUIRED,
        AnvilState.ROUND_DFT_QUEUED,
        AnvilState.ROUND_DFT_DONE,
        AnvilState.ROUND_TRAIN,
        AnvilState.ROUND_TRAINED,
        AnvilState.ROUND_VALIDATED,
    ]
    for s in bootstrap:
        transition(ckpt, s)
    # Run round 0 fully
    for s in round_chain:
        transition(ckpt, s)
    # Loop back into round 1
    transition(ckpt, AnvilState.ROUND_CANDIDATES, advance_round=True)
    assert ckpt.round == 1
    # Walk round 1 to ROUND_VALIDATED
    for s in round_chain[1:]:
        transition(ckpt, s)
    # Then finalize
    transition(ckpt, AnvilState.FINAL_COMPILE)
    transition(ckpt, AnvilState.FINAL_VALIDATE)
    transition(ckpt, AnvilState.DONE)
    assert ckpt.state == AnvilState.DONE


def test_hash_helpers(tmp_path: Path) -> None:
    h1 = hash_bytes(b"hello")
    assert len(h1) == 64
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    assert hash_file(f) == h1
