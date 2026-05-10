"""Integration smoke: full 3-pool bootstrap on amareln Si.

Runs anvil bootstrap with all three pools (A entropy MD + B reform-relax +
C anchor strain) + Tier-2 validation seeds, submits a small VASP array,
waits for convergence, then runs bootstrap_dft_done to collect train.xyz.

Tiny scope: 1 phase × 2 P × 1 T → fast to validate the wiring without
spending much DFT.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    from anvil.orchestrator import Orchestrator
    from anvil.ml.foundation import FoundationRegistry
    from ase.io import read, write as ase_write

    # 1. Build a 16-atom Si supercell seed
    primitive = read("/scratch/lz432/mlip_Si_pbe/data/si_seeds/si_diamond.vasp")
    supercell = primitive.repeat([2, 2, 2])
    sc_path = "/scratch/lz432/anvil_smoke_3pool/si_supercell.vasp"
    Path(sc_path).parent.mkdir(parents=True, exist_ok=True)
    ase_write(sc_path, supercell, format="vasp", vasp5=True, direct=True)

    # 2. yaml — tiny scope (1 phase × 2 P × 1 T)
    cfg_path = "/scratch/lz432/anvil_smoke_3pool/si.yaml"
    Path(cfg_path).write_text(f"""\
system: si_3pool
elements: [Si]
seeds:
  from_local: [{sc_path}]
dft:
  functional: pbe
  encut: 400
  kspacing: 0.5
regime:
  pressures_gpa: [0, 10]
  temperatures_k: [1000]
budget:
  max_dft_calls: 60
  max_walltime_h: 2
generation_quotas:
  pool_a_entropy: 20
  pool_b_reform: 5
  pool_c_anchor: 4
entropy_md:
  k_factors: [2]
  modes: [per_atom]
  md_steps: 100
  fp_cutoff: 5.0
  fp_natx: 50
  regularization: 0.001
  min_entropy_gain: 0.0
reform_relax:
  mode: stability
  pyxtal_n_per_round: 5
  perturb_rattle: 0.3
  perturb_cell: 0.05
  relax_max_steps: 50
  relax_fmax: 0.05
  spglib_filter: true
anchor:
  strain_factors: [0.95, 1.05]
""")

    # 3. Load foundation MLIP once and pass to orchestrator
    print("Loading Allegro-OAM-L...")
    t0 = time.time()
    foundation = FoundationRegistry.make_calc(
        "allegro-oam-l-foundation", cluster="amareln", device="cuda",
        elements=["Si"],
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    # 4. Bootstrap and submit Slurm DFT array
    orch = Orchestrator.from_config_path(
        cfg_path, cluster="amareln",
        scratch_root="/scratch/lz432",
    )
    print(f"run_dir: {orch.run_dir}")
    t0 = time.time()
    orch.bootstrap(do_submit=True, foundation_calc=foundation)
    print(f"bootstrap_dft_queued completed in {time.time()-t0:.1f}s; "
          f"state={orch.ckpt.state.value}")

    import json
    manifest = json.loads((orch.run_dir / "bootstrap_manifest.json").read_text())
    for label, dirs in manifest.items():
        print(f"  {label}: {len(dirs)} struct dirs")
    print(f"  pending_jobs: {len(orch.ckpt.pending_job_ids)}")

    # 5. Wait for VASP jobs to converge (poll squeue every 30s, max 30 min)
    from anvil.hpc.slurm import poll, queue_size
    deadline = time.time() + 30 * 60
    while time.time() < deadline:
        states = poll(orch.ckpt.pending_job_ids)
        n_terminal = sum(1 for s in states.values()
                         if s in ("COMPLETED", "FAILED", "CANCELLED",
                                  "TIMEOUT", "UNKNOWN"))
        if n_terminal == len(states):
            break
        time.sleep(30)
    print(f"\nDFT polling done; states: "
          f"{dict((s, sum(1 for v in states.values() if v == s)) for s in set(states.values()))}")

    # 6. Collect → train.xyz / val.xyz
    t0 = time.time()
    orch.bootstrap_dft_done()
    print(f"\ncollect completed in {time.time()-t0:.1f}s; "
          f"state={orch.ckpt.state.value}")
    report = json.loads((orch.run_dir / "bootstrap_collect_report.json").read_text())
    print(f"  train.xyz: {report['train_collected']} ok / {report['train_failed']} fail")
    print(f"  val.xyz:   {report['val_collected']} ok / {report['val_failed']} fail")
    return 0 if report["train_collected"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
