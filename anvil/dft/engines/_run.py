"""Compute-node entry point: label one struct dir in-process.

Slurm scripts written by the QEpy and ASE-calculator engines end in

    [mpirun -n N] python -m anvil.dft.engines._run <struct_dir>

This module rebuilds the engine from `<struct_dir>/.anvil_meta.json` — which
records the engine name and its options — and calls `engine.run_local()`,
which writes `anvil_result.xyz`. That file is what `engine.collect()` reads,
so the batch path and the in-process path produce identical data.
"""

from __future__ import annotations

import sys
from pathlib import Path

from anvil.dft.engines import engine_from_struct_dir


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m anvil.dft.engines._run <struct_dir>", file=sys.stderr)
        return 2
    struct_dir = Path(argv[0]).resolve()
    if not struct_dir.is_dir():
        print(f"not a directory: {struct_dir}", file=sys.stderr)
        return 2

    engine = engine_from_struct_dir(struct_dir)
    atoms = engine.run_local(struct_dir)
    print(
        f"[{engine.name}] {struct_dir.name}: "
        f"E = {atoms.get_potential_energy():.6f} eV, {len(atoms)} atoms"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via Slurm
    raise SystemExit(main())
