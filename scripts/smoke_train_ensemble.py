"""Integration smoke: train K=3 Allegro fine-tunes from a previous Anvil run.

Uses the train.xyz / val.xyz produced by smoke_3pool_bootstrap.py
(state=BOOTSTRAP_DFT_DONE), walks the orchestrator through
val_dft_done → bootstrap_train → poll → bootstrap_trained.

Tiny scope: 18 train + 4 val structures (very small dataset, will saturate
quickly) — verifies the wiring, not the model quality. Time budget per
member: 30 min.

Usage on amareln:
    python /home/lz432/Anvil/scripts/smoke_train_ensemble.py <run_dir>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main(run_dir_arg: str | None = None) -> int:
    from anvil.orchestrator import Orchestrator
    from anvil.hpc.slurm import poll

    if run_dir_arg is None:
        # Find the most recent si_3pool run dir
        candidates = sorted(
            Path("/scratch/lz432/anvil_runs/si_3pool").iterdir(),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            print("No si_3pool run dirs found. Run smoke_3pool_bootstrap first.")
            return 1
        run_dir_arg = str(candidates[-1])
    print(f"Resuming from {run_dir_arg}")

    orch = Orchestrator.resume(run_dir_arg)
    print(f"State: {orch.ckpt.state.value}")

    if orch.ckpt.state.value == "bootstrap_dft_done":
        orch.val_dft_done()
        print(f"  → {orch.ckpt.state.value}")

    if orch.ckpt.state.value == "val_dft_done":
        # Tiny time budget for the smoke
        orch.bootstrap_train(time="00:30:00")
        print(f"  → {orch.ckpt.state.value}; submitted "
              f"{len(orch.ckpt.pending_job_ids)} training jobs: "
              f"{orch.ckpt.pending_job_ids}")

    # Poll until all training jobs terminate (max 60 min)
    deadline = time.time() + 60 * 60
    last_states = {}
    while time.time() < deadline:
        states = poll(orch.ckpt.pending_job_ids)
        n_done = sum(1 for s in states.values()
                     if s in ("COMPLETED", "FAILED", "CANCELLED",
                              "TIMEOUT", "UNKNOWN"))
        if states != last_states:
            print(f"  poll: {dict((s, sum(1 for v in states.values() if v == s)) for s in set(states.values()))}")
            last_states = states
        if n_done == len(states):
            break
        time.sleep(45)
    print(f"\nTraining poll done.")

    # Move to BOOTSTRAP_TRAINED — gather compiled models
    orch.bootstrap_trained()
    print(f"  → {orch.ckpt.state.value}")
    print(f"  ensemble member hashes: {orch.ckpt.ensemble_hashes}")

    import json
    compiled = json.loads(
        (Path(run_dir_arg) / "ensemble_round0_compiled.json").read_text()
    )["compiled_paths"]
    print(f"  compiled models: {compiled}")
    return 0 if compiled else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
