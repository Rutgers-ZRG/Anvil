"""Pool A entropy-MD integration smoke test for amareln.

Runs a tiny (1 phase × 1 P × 1 T × 1 k × 1 mode) grid using the Allegro-OAM-L
foundation MLIP on a Si diamond seed. Verifies that Pool A actually generates
FP-novel candidates with real MLIP physics.

Usage (on amareln, GPU node):
    sbatch scripts/smoke_pool_a.sh

The wrapper script sets PYTHONPATH and runs this.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    from anvil.al.generation.md_stratified import (
        StratifiedMDConfig,
        generate_pool_a,
    )
    from anvil.ml.foundation import FoundationRegistry

    seed_path = "/scratch/lz432/mlip_Si_pbe/data/si_seeds/si_diamond.vasp"
    fp_state = Path("/scratch/lz432/anvil_smoke_pool_a/fp_state.npz")
    fp_state.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Allegro-OAM-L foundation on cuda...")
    t0 = time.time()
    calc = FoundationRegistry.make_calc(
        "allegro-oam-l-foundation", cluster="amareln", device="cuda",
        elements=["Si"],
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Build a 16-atom supercell from the primitive — the 2-atom primitive is
    # too symmetric for FP-entropy to produce meaningful gain (all atoms
    # equivalent → zero FP variance).
    from ase.io import read
    primitive = read(seed_path)
    supercell = primitive.repeat([2, 2, 2])
    sc_path = "/scratch/lz432/anvil_smoke_pool_a/si_supercell.vasp"
    from ase.io import write as ase_write
    ase_write(sc_path, supercell, format="vasp", vasp5=True, direct=True)
    print(f"Built {len(supercell)}-atom supercell at {sc_path}")

    cfg = StratifiedMDConfig(
        phases={"si_diamond": sc_path},
        pressures_gpa=[0.0, 10.0],
        temperatures_k=[1000.0],
        k_factors=[2.0],
        modes=["per_atom"],
        md_steps=200,
        md_dt_fs=1.0,
        n_snapshots=10,
        fp_cutoff=5.0,
        fp_natx=50,
        regularization=1e-3,
        min_entropy_gain=0.0,        # accept anything with non-negative gain
        relax_fmax=0.05,
        relax_max_steps=50,
        use_local_entropy=False,
    )

    t0 = time.time()
    candidates, ds = generate_pool_a(
        cfg, base_calc=calc,
        n_target=None,
        fp_dataset_path=fp_state,
        verbose=True,
    )
    print(f"\n=== Pool A smoke results ===")
    print(f"Wallclock:       {time.time()-t0:.1f} s")
    print(f"Candidates:      {len(candidates)}")
    print(f"FP dataset size: {ds.count}")
    if candidates:
        for k in ("phase", "pressure_gpa", "temperature_k", "k_factor",
                  "entropy_gain", "sample_kind"):
            sample = [c.info.get(k) for c in candidates[:5]]
            print(f"  first 5 {k}: {sample}")
    return 0 if candidates else 1


if __name__ == "__main__":
    sys.exit(main())
