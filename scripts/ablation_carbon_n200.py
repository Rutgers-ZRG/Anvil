"""Carbon ablation feasibility round (PI option b): N=200 × 4 configs.

Tests the four pool combinations on the same ~800 r2SCAN+rVV10 carbon DFT
budget the previous paper used:

    config         | Pool A | Pool B | Pool C | total
    A only         |  200   |   0    |   0    |  200
    A+B            |  160   |  40    |   0    |  200
    A+C            |  160   |   0    |  40    |  200
    A+B+C (full)   |  140   |  30    |  30    |  200

Each row:
  1. Anvil bootstrap → 200 carbon r2SCAN+rVV10 DFT jobs
  2. Collect → train.xyz
  3. Train K=3 Allegro fine-tunes
  4. Eval against the previous paper's broad_indtest_carbon.xyz (75 cells)

The eval uses the existing labeled broad-indtest from
docs/results_carbon_broad_multiseed.npz reference set, so no new validation
DFT is needed.

Submitted as a long-running Slurm daemon on a CPU node (see .sh wrapper).
Drives the four rows sequentially with polling between submit/collect/train.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SEED_BASE = "/scratch/lz432/allegro_r2scan_finetune/data/mp_seeds"
SEEDS = {
    "diamond":     "mp-66_C_Fd-3m.vasp",
    "graphite":    "mp-48_C_P6_3_mmc.vasp",
    "lonsdaleite": "mp-47_C_P6_3_mmc.vasp",
    "mcarbon":     "mp-1080826_C_C2_m.vasp",
    "cR32":        "mp-169_C_R-3m.vasp",
}

BROAD_INDTEST = "/scratch/lz432/allegro_r2scan_finetune/broad_indtest_carbon.xyz"

ABLATION_ROWS = {
    "A_only":   {"pool_a_entropy": 200, "pool_b_reform":  0, "pool_c_anchor":  0,
                 "skip": ("B", "C")},
    "A_plus_B": {"pool_a_entropy": 160, "pool_b_reform": 40, "pool_c_anchor":  0,
                 "skip": ("C",)},
    "A_plus_C": {"pool_a_entropy": 160, "pool_b_reform":  0, "pool_c_anchor": 40,
                 "skip": ("B",)},
    "full":     {"pool_a_entropy": 140, "pool_b_reform": 30, "pool_c_anchor": 30,
                 "skip": ()},
}


def _ensure_supercells(out_dir: Path, target_atoms: int = 16) -> dict[str, str]:
    """Build 16-atom supercells of each carbon seed. Returns phase→path."""
    from ase.io import read, write as ase_write
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_paths: dict[str, str] = {}
    for phase, fname in SEEDS.items():
        src = Path(SEED_BASE) / fname
        if not src.exists():
            print(f"  skipping {phase} (seed not on amareln: {src})")
            continue
        atoms = read(str(src))
        n = len(atoms)
        if n >= target_atoms:
            sc = atoms
        else:
            import math
            factor = max(1, int(math.ceil((target_atoms / n) ** (1 / 3))))
            sc = atoms.repeat([factor, factor, factor])
        dst = out_dir / f"{phase}.vasp"
        ase_write(str(dst), sc, format="vasp", vasp5=True, direct=True)
        phase_paths[phase] = str(dst)
        print(f"  {phase}: {n} atoms → {len(sc)} atoms → {dst}")
    return phase_paths


def _row_yaml(row_name: str, row: dict, phase_paths: dict[str, str],
              budget_dft: int, scratch_root: str) -> str:
    """Write the per-row Anvil yaml. Returns path."""
    yaml_text = f"""\
system: carbon_ablation_{row_name}
elements: [C]
seeds:
  from_local:
"""
    for path in phase_paths.values():
        yaml_text += f"    - {path}\n"
    yaml_text += f"""\
dft:
  functional: r2scan_rvv10
  encut: 600
  kspacing: 0.25
regime:
  pressures_gpa: [0, 10, 30, 60, 100]
  temperatures_k: [1000, 3000]
budget:
  max_dft_calls: {budget_dft}
  max_walltime_h: 24
generation_quotas:
  pool_a_entropy: {row["pool_a_entropy"]}
  pool_b_reform: {row["pool_b_reform"]}
  pool_c_anchor: {row["pool_c_anchor"]}
entropy_md:
  k_factors: [2, 5]
  modes: [per_atom]
  md_steps: 800
  fp_cutoff: 5.0
  fp_natx: 50
  regularization: 0.001
  min_entropy_gain: 0.005
reform_relax:
  mode: stability
  pyxtal_n_per_round: 8
  perturb_rattle: 0.3
  perturb_cell: 0.05
  relax_max_steps: 80
  relax_fmax: 0.05
  spglib_filter: true
anchor:
  strain_factors: [0.95, 0.97, 1.03, 1.05]
training:
  ensemble_size: 3
"""
    cfg_path = Path(scratch_root) / f"carbon_ablation_n200/{row_name}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml_text)
    return str(cfg_path)


def _wait_jobs(orch, *, max_minutes: int = 24 * 60, poll_s: float = 60.0,
               label: str = "") -> dict[str, str]:
    """Poll orch.ckpt.pending_job_ids until all terminate. Returns {jid: state}."""
    from anvil.hpc.slurm import poll
    deadline = time.time() + max_minutes * 60
    last = {}
    while time.time() < deadline:
        st = poll(orch.ckpt.pending_job_ids)
        n_term = sum(1 for s in st.values()
                     if s in ("COMPLETED", "FAILED", "CANCELLED",
                              "TIMEOUT", "UNKNOWN"))
        if st != last:
            counts = {s: sum(1 for v in st.values() if v == s) for s in set(st.values())}
            print(f"  [{label}] {counts}", flush=True)
            last = st
        if n_term == len(st):
            return st
        time.sleep(poll_s)
    print(f"  [{label}] WARNING: deadline reached, returning current states")
    return last


def run_ablation_row(row_name: str, row: dict,
                     phase_paths: dict[str, str], scratch_root: str,
                     foundation_calc) -> dict:
    """Run one ablation row end-to-end: bootstrap → collect → train → eval."""
    from anvil.orchestrator import Orchestrator
    from anvil.ml.ensemble import EnsembleCalculator
    from anvil.validation.tier2_indtest import evaluate_tier2

    print(f"\n{'='*60}")
    print(f"ROW {row_name}: A={row['pool_a_entropy']} B={row['pool_b_reform']} "
          f"C={row['pool_c_anchor']} skip={row['skip']}")
    print(f"{'='*60}")

    cfg_path = _row_yaml(row_name, row, phase_paths,
                          budget_dft=row["pool_a_entropy"]
                                     + row["pool_b_reform"]
                                     + row["pool_c_anchor"] + 100,
                          scratch_root=scratch_root)

    orch = Orchestrator.from_config_path(
        cfg_path, cluster="amareln", scratch_root=scratch_root,
    )
    print(f"  run_dir: {orch.run_dir}")

    # 1. Bootstrap (3-pool gen + DFT submit)
    t0 = time.time()
    orch.bootstrap(do_submit=True, foundation_calc=foundation_calc,
                   skip_pools=row["skip"])
    print(f"  bootstrap submitted; {len(orch.ckpt.pending_job_ids)} VASP jobs "
          f"({time.time()-t0:.0f}s)")

    # 2. Wait for VASP to converge
    _wait_jobs(orch, label=f"{row_name}-dft")

    # 3. Collect
    orch.bootstrap_dft_done()
    orch.val_dft_done()
    print(f"  state: {orch.ckpt.state.value}")

    # 4. Submit training
    orch.bootstrap_train(do_submit=True, time="08:00:00")
    print(f"  training submitted: {orch.ckpt.pending_job_ids}")

    _wait_jobs(orch, label=f"{row_name}-train", poll_s=60.0)

    # 5. Collect compiled
    orch.bootstrap_trained()
    print(f"  state: {orch.ckpt.state.value}")

    # 6. Eval on broad-indtest from previous paper
    print(f"  evaluating on broad-indtest ({BROAD_INDTEST})...")
    compiled = json.loads(
        (orch.run_dir / "ensemble_round0_compiled.json").read_text()
    )["compiled_paths"]
    ens = EnsembleCalculator(compiled, device="cuda")
    metrics = evaluate_tier2(ens, BROAD_INDTEST, thresholds={
        "E_MAE": 1e9, "F_MAE": 1e9, "S_MAE": 1e9,    # disable threshold check
    })
    metrics["row_name"] = row_name
    metrics["pool_quotas"] = {k: row[k] for k in
                               ("pool_a_entropy", "pool_b_reform", "pool_c_anchor")}
    metrics["run_dir"] = str(orch.run_dir)
    (orch.run_dir / "broad_indtest_eval.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(f"  → E_MAE={metrics['E_MAE']:.2f} F_MAE={metrics['F_MAE']:.1f} "
          f"S_MAE={metrics['S_MAE']:.2f}")
    return metrics


def main() -> int:
    from anvil.ml.foundation import FoundationRegistry

    scratch_root = "/scratch/lz432"
    seeds_dir = Path(scratch_root) / "carbon_ablation_n200" / "seeds"
    print("Building 16-atom carbon supercells...")
    phase_paths = _ensure_supercells(seeds_dir, target_atoms=16)
    if not phase_paths:
        print("FATAL: no carbon seeds available")
        return 1

    print(f"\nLoading Allegro-OAM-L on cuda...")
    foundation = FoundationRegistry.make_calc(
        "allegro-oam-l-foundation", cluster="amareln", device="cuda",
        elements=["C"],
    )
    print(f"  loaded")

    summary = {}
    for row_name, row in ABLATION_ROWS.items():
        try:
            metrics = run_ablation_row(
                row_name, row, phase_paths, scratch_root,
                foundation_calc=foundation,
            )
            summary[row_name] = {
                "E_MAE": metrics["E_MAE"], "F_MAE": metrics["F_MAE"],
                "S_MAE": metrics["S_MAE"], "n_struct": metrics["n_struct"],
                "subset": metrics.get("subset", {}),
                "run_dir": metrics["run_dir"],
            }
        except Exception as exc:
            print(f"\n!!! ROW {row_name} FAILED: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            summary[row_name] = {"error": str(exc)}

        # Persist running summary so partial results survive a crash
        Path(f"{scratch_root}/carbon_ablation_n200/summary.json").write_text(
            json.dumps(summary, indent=2)
        )

    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY (broad-indtest, 75 cells)")
    print(f"{'='*60}")
    print(f"{'row':<12s} {'E_MAE':>8s} {'F_MAE':>8s} {'S_MAE':>8s}  "
          f"(meV/{{at,Å,Å³}})")
    for row_name in ABLATION_ROWS:
        m = summary.get(row_name, {})
        if "error" in m:
            print(f"{row_name:<12s}  FAILED: {m['error']}")
        else:
            print(f"{row_name:<12s} {m['E_MAE']:>7.2f} {m['F_MAE']:>7.1f} "
                  f"{m['S_MAE']:>7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
