"""Pool B reform-relax integration smoke test for amareln.

Generates 5 perturbed Si seeds + reform-relax with foundation Allegro-OAM-L
base. Verifies that relaxation converges and at least some final cells pass
the spglib symmetry filter.

Usage (on amareln, GPU node):
    sbatch scripts/smoke_pool_b.sh
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    from anvil.al.generation.reform_relax import (
        ReformRelaxConfig,
        generate_pool_b,
    )
    from anvil.ml.foundation import FoundationRegistry

    seed_path = "/scratch/lz432/mlip_Si_pbe/data/si_seeds/si_diamond.vasp"

    print("Loading Allegro-OAM-L on cuda...")
    t0 = time.time()
    calc = FoundationRegistry.make_calc(
        "allegro-oam-l-foundation", cluster="amareln", device="cuda",
        elements=["Si"],
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Build a 16-atom supercell (same lesson from Pool A: primitive too symmetric)
    from ase.io import read, write as ase_write
    primitive = read(seed_path)
    supercell = primitive.repeat([2, 2, 2])
    sc_path = "/scratch/lz432/anvil_smoke_pool_b/si_supercell.vasp"
    Path(sc_path).parent.mkdir(parents=True, exist_ok=True)
    ase_write(sc_path, supercell, format="vasp", vasp5=True, direct=True)
    print(f"Built {len(supercell)}-atom supercell")

    cfg = ReformRelaxConfig(
        mode="stability",
        perturbations_per_seed=5,
        perturb_rattle=0.3,
        perturb_cell=0.05,
        relax_max_steps=80,
        relax_fmax=0.05,
        spglib_filter=True,
        reform_lambda=1.0,
        reform_cutoff=4.0,
        reform_nx=200,
    )

    t0 = time.time()
    candidates = generate_pool_b(
        cfg,
        base_calc=calc,
        seed_phases={"si_diamond": sc_path},
        seed=42,
        verbose=True,
    )
    print(f"\n=== Pool B smoke results ===")
    print(f"Wallclock:       {time.time()-t0:.1f} s")
    print(f"Candidates kept: {len(candidates)} / {cfg.perturbations_per_seed}")
    for i, a in enumerate(candidates[:5]):
        print(f"  [{i}] phase={a.info['phase']} mode={a.info.get('reform_mode')} "
              f"natoms={len(a)}")
        try:
            import spglib
            ds = spglib.get_symmetry_dataset(
                (a.get_cell()[:], a.get_scaled_positions(),
                 a.get_atomic_numbers()), symprec=0.1,
            )
            if ds is not None:
                # spglib >= 2 uses attribute access on a dataclass.
                num = getattr(ds, "number", None)
                if num is None:                          # fallback for older spglib
                    try:
                        num = ds["number"]
                    except (KeyError, TypeError):
                        num = "?"
                print(f"        spglib: SG #{num}")
        except ImportError:
            pass
        except Exception as exc:
            print(f"        spglib lookup failed: {exc}")
    return 0 if candidates else 1


if __name__ == "__main__":
    sys.exit(main())
