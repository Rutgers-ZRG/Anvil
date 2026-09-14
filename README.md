# Anvil

**A**ctive-learning **N**etwork **V**alidation and **I**terative **L**abeling

> Closed-loop active learning for fine-tuning machine-learned interatomic potentials.
> One yaml. One command. A reliable, validated MLIP.

**Status**: v1 code-complete (51 passing tests) — resumable closed-loop
orchestration, three-pool generation, dataset-aware acquisition, Allegro
fine-tuning, and tiered validation. DFT labeling goes through a pluggable
engine layer: VASP + Slurm (default), Quantum ESPRESSO via
[QEpy](https://github.com/EACcodes/qepy) or `pw.x`, or any ASE calculator.
First end-to-end result: a carbon pool-ablation study.

See [`DESIGN.md`](./DESIGN.md) for the v1 architecture, three-pool generation
strategy, validation tiers, and milestones.

The method implemented here is described in:

> Meiyan Wang, Rishi Rao, and Li Zhu,
> *Dataset-aware entropy-maximized active learning for machine-learned
> interatomic potentials*,
> [arXiv:2605.20384](https://arxiv.org/abs/2605.20384) (2026).

## Quick start

```bash
pip install -e .
anvil submit --config configs/examples/carbon.yaml
# ... ~24 h later, depending on DFT budget ...
anvil report carbon
```

## DFT engines

Anvil is not tied to VASP. `dft.engine` picks the code that produces the
labels; `dft.engine_options` holds its settings.

| `dft.engine` | Backend | Install |
|---|---|---|
| `vasp` (default) | VASP + Slurm | VASP binary + POTCARs on the cluster |
| `qe` | Quantum ESPRESSO — in-process via QEpy (`mode: qepy`) or batch `pw.x` (`mode: pwx`) | `pip install anvil-mlip[qe]`, or just a `pw.x` binary for `pwx` |
| `ase` | Any ASE calculator: GPAW, CP2K, Abinit, FHI-aims, xTB... | whatever that calculator needs |

```yaml
# Quantum ESPRESSO instead of VASP — everything else in the config is unchanged
dft:
  engine: qe
  kspacing: 0.25
  engine_options:
    mode: qepy
    pseudo_dir: /path/to/pseudos
    pseudopotentials: {Si: Si.pbe-n-kjpaw_psl.1.0.0.UPF}
    input_data:
      system: {ecutwfc: 60, ecutrho: 480}
```

```yaml
# Any ASE calculator
dft:
  engine: ase
  engine_options:
    calculator: gpaw.GPAW      # import path, or a shortcut like "emt" / "cp2k"
    use_kpts: true
    kwargs: {mode: {name: pw, ecut: 600}, xc: PBE}
```

Runnable examples: [`configs/examples/si_qe.yaml`](./configs/examples/si_qe.yaml),
[`configs/examples/si_ase.yaml`](./configs/examples/si_ase.yaml).
Adding another code means one subclass of `DFTEngine`
(`anvil/dft/engines/base.py`) — write inputs, submit, check convergence,
collect. Every engine returns ASE units and the ASE (compression-negative)
stress convention, so the training data is identical whichever code labeled it.

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
    dft/               — pluggable labeling engines (VASP / QE+QEpy / ASE calc),
                         Slurm runner, convergence + sanity checks
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
- [ASE](https://wiki.fysik.dtu.dk/ase/) — atoms, calculators, and the I/O the
  engine layer is built on.
- [QEpy](https://github.com/EACcodes/qepy) — Quantum ESPRESSO as a Python
  object, used by the `qe` engine.
- Karabin & Perez, *J. Chem. Phys.* **153**, 094110 (2020); Subramanyam &
  Perez, *npj Comput. Mater.* **11**, 218 (2025).

## Citation

If you use Anvil, please cite:

```bibtex
@misc{wang2026anvil,
  title  = {Dataset-aware entropy-maximized active learning for
            machine-learned interatomic potentials},
  author = {Wang, Meiyan and Rao, Rishi and Zhu, Li},
  year   = {2026},
  eprint = {2605.20384},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2605.20384}
}
```
