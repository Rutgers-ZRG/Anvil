"""Integration smoke: load the trained Si ensemble + evaluate on val.xyz.

Walks Orchestrator from BOOTSTRAP_TRAINED → ROUND_CANDIDATES via the new
round_validated() method. Computes E/F/S MAE on the 4 Tier-2 cells from
the previous bootstrap.

Tiny dataset (4 strain probes); MAEs on this set are not meaningful as a
quality assessment, but the pipe should produce numeric metrics.

Usage on amareln:
    python /home/lz432/Anvil/scripts/smoke_tier2_eval.py [<run_dir>]
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(run_dir_arg: str | None = None) -> int:
    from anvil.orchestrator import Orchestrator

    if run_dir_arg is None:
        candidates = sorted(
            Path("/scratch/lz432/anvil_runs/si_3pool").iterdir(),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            print("No si_3pool run dirs found.")
            return 1
        run_dir_arg = str(candidates[-1])
    print(f"Resuming from {run_dir_arg}")

    orch = Orchestrator.resume(run_dir_arg)
    print(f"State: {orch.ckpt.state.value}")

    if orch.ckpt.state.value != "bootstrap_trained":
        print(f"  expected bootstrap_trained, got {orch.ckpt.state.value}")
        return 2

    metrics = orch.round_validated(device="cuda")
    print(f"\n=== Tier-2 evaluation ===")
    print(f"  state: {orch.ckpt.state.value}")
    print(f"  n_struct: {metrics['n_struct']}")
    print(f"  E_MAE: {metrics['E_MAE']:.3f} meV/atom")
    print(f"  F_MAE: {metrics['F_MAE']:.2f} meV/Å")
    print(f"  S_MAE: {metrics['S_MAE']:.3f} meV/Å³")
    print(f"  passed: {metrics['passed']}")
    if not metrics['passed']:
        print(f"  failed thresholds: {metrics['failed_thresholds']}")
    print(f"  by subset:")
    for k, v in metrics.get("subset", {}).items():
        print(f"    {k}: E={v['E_MAE']:.2f}  F={v['F_MAE']:.1f}  "
              f"S={v['S_MAE']:.2f}  (n={v['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
