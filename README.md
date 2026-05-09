# Anvil

> Closed-loop active learning for fine-tuning machine-learned interatomic potentials.
> One yaml. One command. A reliable, validated MLIP.

**Status**: design phase. Skeleton-only. No algorithm code yet.

See [`DESIGN.md`](./DESIGN.md) for the v1 architecture, three-pool generation
strategy, validation tiers, and milestones. Cut code only after the design is
approved.

## Quick start (planned UX, not yet implemented)

```bash
pip install -e .
anvil submit --config configs/examples/carbon.yaml
# ... ~24 h later, depending on DFT budget ...
anvil report carbon
```

## Method

Anvil extends the dataset-aware entropy-maximized sampling line of work
(Karabin & Perez 2020; Subramanyam & Perez 2025) by combining three
complementary structure-generation pools per active-learning round:

| Pool | Generates | Purpose |
|---|---|---|
| **A. Per-regime entropy MD** | FP-novel, off-equilibrium | "diverse, exploratory" — workhorse for energy |
| **B. Reform-relax** | ordered crystals near local minima | "stable phases" — closes the carbon F-gap on strain probes |
| **C. Anchor** | exact user-regime structures | "task-aligned" — ensures support across regime grid |

Acquisition is a thin quota merge with FP-distance dedupe at pool boundaries.
The closed loop adds a foundation-MLIP ensemble whose uncertainty informs (in
v2) candidate ranking.

## Repository

```
anvil/
    cli.py             — submit / resume / status / report / cancel / daemon
    orchestrator.py    — state machine driving the AL loop
    config.py          — yaml schema + validation
    state.py           — checkpoint format + transitions
    dft/               — VASP input/output + Slurm runner + convergence checks
    ml/                — foundation MLIP registry, K-seed ensemble trainer, compile
    al/
        generation/    — five generators (local + global entropy, stratified MD,
                         reform-relax, anchor)
        acquisition.py — quota merge + FP-distance dedupe
        stop.py        — stop-criterion checker
    validation/        — three tiers (physics / mini broad-indtest / full)
    hpc/               — Slurm wrappers + cluster registry (amareln, amarel3)
    report/            — HTML report + plots

configs/
    defaults.yaml
    functionals/       — INCAR templates per DFT functional
    examples/          — carbon, si, nacl

tests/
docs/
```

## License

MIT. See [`LICENSE`](./LICENSE).

## Acknowledgments

Builds on:
- [reformpy](https://github.com/Rutgers-ZRG/ReformPy) — Reform_Calculator,
  EntropyMaximizingCalculator, libfp wrappers.
- [libfp](https://github.com/Rutgers-ZRG/libfp) — atomic fingerprints with
  analytic gradients.
- [nequip](https://github.com/mir-group/nequip) and Allegro — equivariant
  neural-network MLIPs.
- Karabin & Perez, *J. Chem. Phys.* **153**, 094110 (2020); Subramanyam &
  Perez, *npj Comput. Mater.* **11**, 218 (2025).
