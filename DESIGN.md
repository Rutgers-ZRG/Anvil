# Anvil — Design Doc (v1)

> *Closed-loop active learning for fine-tuning MLIPs on user-specified physical regimes.*
> *One yaml. One command. A reliable, validated MLIP.*

**Status**: design phase. No code yet. Revise this document before implementing.
**Author**: mlip-trainer (Z-group), under PI direction.
**Date**: 2026-05-07.

---

## 1. Goals and non-goals

### 1.1 Goals (v1)

1. **Single-command turnkey**: `anvil submit --config carbon.yaml` starts a
   closed-loop run; `anvil report carbon` gives a final go/no-go signal and a
   deployment-ready compiled MLIP.
2. **Reliable**: built-in validation suite catches catastrophic failures
   (de-stabilized phases, broken EOS) before delivering the model. Every
   delivered model carries an explicit "supported regime" box.
3. **Dataset-aware entropy with regime + stable-phase coverage**:
   the existing local + global entropy method already wins on energy across
   carbon / Si / NaCl. The remaining gap, measured on the broad-indtest, is
   carbon force prediction on near-zero-force ordered crystals — a regime
   that pure FP-novelty entropy systematically avoids. Anvil adds two
   complementary pools (reform-relax for ordered phases, anchor for the
   user-specified regime) on top of the existing entropy generator. The
   method-paper claim is the **ablation**: each pool's marginal contribution
   to the broad-indtest MAE, with Pool A alone as the baseline (this paper).
4. **HPC-aware**: amareln + amarel3 (Slurm), checkpointed, idempotent restarts,
   never violates login-node policy.
5. **Reproducible**: every model + every pool + every validation set is content
   hashed. `anvil status` is the source of truth.

### 1.2 Non-goals (deferred to v2+)

- Other DFT codes (QE, CP2K, VASP-only in v1).
- Other foundation MLIPs (MACE-MP, MatterSim — v2).
- Other clusters (general Slurm, PBS, cloud — v2).
- Heterogeneous-foundation ensembles (different architectures voting — v2).
- Generative-model candidate sampling (flow / VAE — v3).
- Web UI / dashboard (CLI + HTML report only in v1).
- Multi-element with sparse element coverage by foundation model (v2).

---

## 2. The single user input

```yaml
# carbon.yaml — minimum viable config
system: carbon
elements: [C]            # used to validate foundation-model element coverage

seeds:
  from_mp: [mp-66, mp-48, mp-47, mp-1080826, mp-169]
  # Or: from_local: [./poscars/*.vasp]

dft:
  functional: r2scan_rvv10   # built-in INCAR template; see configs/functionals/
  encut: 600
  kspacing: 0.25             # default: a/x for x-density
  # extra INCAR overrides via dft.extra_incar: {NPAR: 4, ...}

regime:                       # the "where do I want it accurate" box
  pressures_gpa: [0, 10, 30, 60, 100]
  temperatures_k: [300, 1000, 3000]

budget:
  max_dft_calls: 800
  max_walltime_h: 48

# Everything below has defaults; user overrides only when they need to.
```

Defaults that the framework supplies:

- foundation\_model: `allegro-oam-l-foundation`
- ensemble\_size: 3
- bootstrap\_size: 100
- per\_round\_dft: 60
- acquisition: `stratified_sigma_F`
- ensemble\_loss\_weights: `1:1:0.01`  (E:F:S)
- learning\_rate: `5e-5`, patience 30, max epochs 200
- compile\_target: `torchscript-ase`
- validation\_tier: 2
- validation\_thresholds: `{E_MAE: 10, F_MAE: 100, S_MAE: 10}` (units meV/{at,Å,Å³})

Every default is overridable via the yaml, but the user is never *required* to
set ML hyperparameters.

---

## 3. The closed-loop algorithm

### 3.1 Three-pool generation, dataset-aware

The structure-generation step combines three complementary pools, each
covering a region of configuration space the others miss:

| Pool                         | Driver                                                                                                | Generates                                                     | Region of config space                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------- |
| **A. Per-regime entropy MD** | reformpy local entropy + ported global entropy, per (phase, P, T, k\_factor, mode)                    | FP-novel, off-equilibrium, high-force                         | "diverse, exploratory" — workhorse for energy              |
| **B. Reform-relax**          | reformpy `Reform_Calculator` + foundation MLIP, *minimizing* FP-distance to nearest seed-phase target | ordered crystals, near local minima, low-force, high symmetry | "stable phases" — closes the carbon F-gap on strain probes |
| **C. Anchor**                | foundation MLIP only: relax + isotropic strain probes + short Langevin MD at each user (P, T)         | exact user-regime structures, low-force, ordered              | "task-aligned" — ensures support across regime grid        |

The three are run in parallel each round; acquisition is a thin quota merge
with FP-distance dedupe at the boundaries (no σ\_F-based selection in v1).

### 3.2 Pool A — per-regime entropy MD (existing method, regime-stratified)

Continues the local + global entropy lineage of Karabin/Perez 2020 +
Subramanyam/Perez 2025. The structure-generation logic ports two existing
files from `/Users/li/dev/RA/mlip-active-learn/`:

- `global_entropy_calculator.py` (313 LOC) — `FingerprintDataset` + `GlobalEntropyCalculator`,
  pure NumPy + libfp, ASE-Calculator conformant, no MPI. Direct port to
  `anvil/al/generation/global_entropy.py`.
- `generate_structures.py` (relevant slices) — the per-condition stratification
  loop over (T × k\_factor × P × phase × mode). Ported to
  `anvil/al/generation/md_stratified.py`.

For each AL round, Pool A runs short entropy-driven MD trajectories
**independently per (phase, P, T, k\_factor, mode)** rather than as one global
MD. Each cell's `FingerprintDataset` is shared across phases / pressures /
temperatures so global novelty is preserved, but starting from a relaxed
phase at the target (P, T) ensures every regime gets entropy-novel candidates
representative of that regime.

```
For phase × P × T × k_factor × mode:
  relaxed = foundation_relax(phase, P)
  thermalize at T
  md_steps with EntropyMaximizingCalc (local, reformpy)
            and GlobalEntropyCalc (global, ported)
  accept candidates via FingerprintDataset.entropy_gain > min_gain
```

### 3.3 Pool B — reform-relax (CRISP-style, FP-targeted)

Drives a small set of initial structures toward ordered phases via
`reformpy.Reform_Calculator` + foundation MLIP base.

```
Initial structures (mix per round):
  - perturbed seed phases (rattle 0.3 Å + cell ±5%)        (recovery)
  - random pyXtal cells at the system composition           (discovery)
  - low-energy, high-F snapshots from Pool A                (refinement)

Target FP per candidate (default mode (a) — stability):
  fp_target = nearest seed-phase FP at the candidate's current pressure

Opt-in mode (b) — discovery (yaml: reform_relax.mode: discovery):
  fp_target = farthest seed-phase FP    (drives toward novel basins)

Relaxation: BFGS / FIRE with
  F_total = F_foundation - λ · J^T · ∇_fp ||fp - fp_target||²
until fmax < 0.05 eV/Å or 200 steps.

Sample:
  - the final relaxed cell                            (the crystal-like structure)
  - 1-2 intermediate snapshots near convergence       (low-F but not strict minimum)
  - optional spglib symmetry filter (drop P1, keep ≥ Pmnm)
```

Pool B has a built-in dataset-aware property: candidates entering the pool
are FP-distance pruned against pool ∪ A ∪ C, so even though the relaxation
*targets* phase-like structures, no candidate is added if its FP is already
represented. Pool B fills the **ordered-crystal regime that pure FP-novelty
entropy avoids by construction**.

### 3.4 Pool C — anchor (regime alignment)

Foundation-MLIP only; no entropy or FP machinery. Designed to populate the
zero-force / near-equilibrium regime that the indtest probes.

```
For phase in seed_phases:
  For P in user.pressures_gpa:
    foundation_relax(phase, P)                        → 1 equilibrium cell
    apply isotropic strain {0.95, 0.97, 1.03, 1.05}   → 4 strain probes
    For T in user.temperatures_k:
      Langevin MD 50–100 fs, 5 snapshots              → 5 thermal cells

Per (phase, P): 1 + 4 + 5·|T| structures.
Sub-sample to pool-budget quota.
```

The strain factors {0.95, 0.97, 1.03, 1.05} are smaller than the broad-indtest
{0.92, 1.08} — anchor structures are training data that should stay close to
equilibrium (not the indtest's harder probes).

### 3.5 Closed-loop algorithm

```
Round 0 — bootstrap
  0.1  Pull seed structures (MP API or user POSCARs)
  0.2  Foundation-MLIP relaxation at each (phase, P) in regime grid
  0.3  Generate small Pool C anchor batch                 → ~80 candidates
       Generate small Pool A entropy MD batch             → ~60 candidates
       Generate small Pool B reform-relax bootstrap       → ~40 candidates
  0.4  Build validation set in parallel: Tier-2 strain probes + hot snaps
       (script ported from build_broad_indtest.py)
  0.5  DFT-label N₀ ≈ 180 pool + N_val ≈ 50 validation (parallel array jobs)
  0.6  Train K=3 fine-tunes from different seeds → ensemble {f₁, f₂, f₃}

Round r ≥ 1 — closed loop
  r.1  Pool A: per-regime entropy MD using f̄ as base calc        (~50 DFT)
  r.2  Pool B: reform-relax with target = nearest seed-phase FP   (~20 DFT)
       (or farthest if reform_relax.mode=discovery)
  r.3  Pool C: foundation-MLIP anchor refresh on under-filled bins (~10 DFT)
  r.4  Quota merge → FP-distance dedupe at boundaries              (~80 selected)
  r.5  DFT-label selected; collect; merge into training pool
  r.6  Retrain ensemble (warm-start from previous round; full retrain every 3
       rounds for safety)
  r.7  Tier-1 physics check on f̄:
         - Seed phases stable under MLIP relaxation?
         - EOS V(P) smooth, single-minimum?
         - Phonon ω_min > -5 cm⁻¹?
       If any fail: state machine → PAUSED_PHYSICS, user notified.
  r.8  Compute monitor metrics: validation MAE per round.

Stop when ANY:
  S.1  Budget exhausted (DFT calls or wallclock).
  S.2  Validation MAE plateau 2 consecutive rounds (rel. change < 5%).
  S.3  Tier-1 physics check fails (escalation, not graceful stop).
```

### 3.6 Method-paper claim — pool ablation

The publication claim is the **ablation table**, not a single threshold:

| Pool config                         | Carbon broad-indtest E / F / S                    | NaCl ditto        | Si ditto         |
| ----------------------------------- | ------------------------------------------------- | ----------------- | ---------------- |
| Pool A only (baseline = this paper) | (measured: 36.7 / 96 / 30)                        | (1.0 / 1.6 / 0.4) | (6.9 / 18 / 1.9) |
| Pool A + B                          | predicted: F ↓ on strain                          | unchanged or ↓    | E unchanged or ↑ |
| Pool A + C                          | predicted: F ↓ on strain                          | unchanged         | unchanged        |
| Pool A + B + C (full Anvil)         | predicted: F closes carbon gap, E and S preserved | clean win         | clean win        |

48 runs (4 configs × 3 systems × 4 sizes × 1 seed for ablation, can multi-seed
later). Same DFT budget per row (800), same broad-indtest evaluation set,
same multi-seed protocol as this paper. Each row's marginal contribution
isolates one pool's value.

---

## 4. Module breakdown

### 4.1 `anvil.cli`

```
anvil submit  --config <yaml>     # cold start
anvil resume  <run_id>            # warm start from checkpoint
anvil status  <run_id>            # state machine inspection
anvil report  <run_id>            # generate HTML report
anvil cancel  <run_id>            # graceful shutdown (cancels SLURM jobs)
anvil daemon  <run_id>            # long-running orchestrator (srun job)
```

### 4.2 `anvil.orchestrator`

The state machine. Every transition writes a JSON checkpoint to
`<run_dir>/.anvil/state.json`.

States:

```
INIT → BOOTSTRAP_SEEDS → BOOTSTRAP_DFT_QUEUED → BOOTSTRAP_DFT_DONE →
VAL_DFT_QUEUED → VAL_DFT_DONE → BOOTSTRAP_TRAIN → BOOTSTRAP_TRAINED →
ROUND_r_CANDIDATES → ROUND_r_ACQUIRED → ROUND_r_DFT_QUEUED →
ROUND_r_DFT_DONE → ROUND_r_TRAIN → ROUND_r_TRAINED → ROUND_r_VALIDATED →
{loop or} → FINAL_COMPILE → FINAL_VALIDATE → DONE
```

`resume` reads `state.json`, jumps to the matching state, idempotently re-runs
incomplete steps. Any state can be re-entered without corruption.

### 4.3 `anvil.dft.vasp`

- INCAR templates per functional (`r2scan_rvv10`, `pbe`, `scan`, `pbe_d3`).
  Templates inherit from PI's existing `*_setup_vasp.py` scripts.
- KPOINTS generator with k-density, max-kpts cap (PI's convention).
- POTCAR auto-resolution from `/home/lz432/apps/PBE64/<element>/POTCAR`.
- Slurm wrapper with throttled submit (queue-cap aware).
- Output validation: convergence + |F| sanity + extxyz collection.
- Idempotency: re-running a struct dir checks for existing converged OUTCAR
  before re-submitting.

### 4.4 `anvil.ml.foundation`

Registry of foundation MLIPs. v1 has Allegro-OAM-L only.

```python
foundation = FoundationRegistry.get("allegro-oam-l")
# Returns a callable that loads the .nequip.pth, returns ASE calc.
foundation.supported_elements  # → set of Z's covered
foundation.checkpoint_path     # → /scratch/lz432/allegro_finetune/...
```

Pre-flight check: framework refuses to start if user's elements aren't all in
`foundation.supported_elements`. Suggests `from_dft_only: true` workaround.

### 4.5 `anvil.ml.ensemble`

K-seed Allegro fine-tune trainer. K=3 by default. Produces an ensemble
calculator that exposes:

- `get_potential_energy()` → mean over K
- `get_forces()` → mean over K
- `get_stress()` → mean over K
- `get_uncertainty()` → {σ\_E, σ\_F, σ\_S}

Each ensemble member trained with same hyperparameters, different `data.seed`.
Warm-start from previous round's checkpoints by default (full restart every 3
rounds for safety).

### 4.6 `anvil.al.generation` (three pools)

```
anvil/al/generation/
    local_entropy.py       # ~50 LOC; thin wrapper over reformpy.EntropyMaximizingCalculator
    global_entropy.py      # ~320 LOC; direct port of mlip-active-learn/global_entropy_calculator.py
                           #   (FingerprintDataset + GlobalEntropyCalculator, NumPy + libfp, no MPI)
    md_stratified.py       # ~250 LOC; per-(phase, P, T, k_factor, mode) MD orchestrator;
                           #   ports condition-grid logic from mlip-active-learn/generate_structures.py
    reform_relax.py        # ~180 LOC; NEW — wraps reformpy.Reform_Calculator with
                           #   FP-target selection (modes: nearest-default | farthest-discovery),
                           #   pyXtal random init, perturbed-seed init, low-F snap from Pool A,
                           #   spglib symmetry filter (optional, drop P1)
    anchor.py              # ~120 LOC; foundation-MLIP relax + isotropic strain probes +
                           #   short Langevin MD at each user (P, T)
```

Each module returns ASE Atoms lists with metadata (`pool: A|B|C`, `phase`,
`pressure_gpa`, `temperature_k`, `sample_kind`, source mode for B). Pool C
also tags `strain_factor` for downstream stress diagnostics.

### 4.7 `anvil.al.acquisition`

```python
class QuotaMergeAcquisition:
    """
    v1 default: simple quota merge of A/B/C with FP-distance dedupe.

    No σ_F-based selection in v1 — model uncertainty UQ is a v2 add-on
    (see §10 lessons).
    """
    def __init__(self, quotas, fp_dedupe_threshold):
        # quotas = {'A': 50, 'B': 20, 'C': 10}
        ...

    def select(self, pool_A, pool_B, pool_C, current_training_pool):
        # 1. Within each pool, rank by intrinsic novelty
        #    (Pool A: entropy_gain; Pool B: low-F + reform endpoint;
        #     Pool C: regime-coverage uniformity)
        # 2. Take top-quota from each
        # 3. FP-distance dedupe across pool boundaries
        # 4. Final FP-distance prune vs current_training_pool
        # 5. Return ≤ sum(quotas) atoms
```

Alternate acquisitions for ablation:

- `PoolAOnlyAcquisition` — baseline (this paper's method)
- `PoolABAcquisition` — A + reform-relax only
- `PoolACAcquisition` — A + anchor only
- `QuotaMergeAcquisition` — A + B + C (default, full Anvil)
- (v2) `SigmaFAcquisition` — adds ensemble UQ as an additional ranking signal

These are pluggable for the ablation studies in the method paper (see §3.6).

### 4.8 `anvil.al.stop`

Stop-criterion checker. Reads monitor metrics from
`<run_dir>/.anvil/monitor.jsonl` and decides at the end of each round whether
to continue.

### 4.9 `anvil.validation`

Three-tier validation suite. See §5.

### 4.10 `anvil.hpc.slurm`

- Cluster registry: `amareln`, `amarel3` (more in v2).
- Each cluster has: scheduler params, conda env path, GPU partition,
  module loads, foundation-model-cache path.
- `cluster_for(host) -> ClusterConfig` auto-detects from hostname.
- Throttled submit with backoff when queue full.
- `sbatch` wrapper that ensures all logs go under `<run_dir>/logs/`.

### 4.11 `anvil.hpc.env`

Shadow-variable pattern for conda activation in Slurm scripts (we hit `$SIZE`
clobber in the paper experiments — codified here so it never recurs).

### 4.12 `anvil.report`

HTML generator. Plots:

- Learning curves of monitor metrics vs round
- σ\_F trajectory per round
- Parity plots (E, F, S) on validation set, final ensemble
- Phase-coverage heatmap: (phase × P × T) grid colored by E\_MAE
- EOS V(P) curves per phase: MLIP vs DFT seed
- Phonon ω plots if Tier-3
- Summary table with PASS/FAIL per check

---

## 5. Validation suite

Built upfront from `regime:` spec; never touches training pool.

### 5.1 Tier 1 — physics checks (free, every round)

After every retrain:

- Each seed phase relaxed with f̄ from initial cell; check basin preserved
  (RMSD of relaxed positions < 0.1 Å vs initial scaled cell, no symmetry break).
- E(V) at 7 volume points around equilibrium; fit Birch-Murnaghan; require
  smooth, single-minimum, B₀ within ±20% of MP reference.
- Phonon ω\_min at Γ for each phase (small displacement); require > -5 cm⁻¹.
- Phase-ordering enthalpy at user pressures (using f̄'s relaxed cells); require
  ordering matches DFT seeds (or foundation MLIP if no DFT seed).

If any fails: state machine → `PAUSED_PHYSICS`. User notified, reviews report.

### 5.2 Tier 2 — mini broad-indtest (default, \~50 DFT calls)

Built once at bootstrap, never enters training:

```
For phase in seed_phases:
  For P in regime.pressures_gpa:
    Foundation-MLIP-relaxed cell
    Apply isotropic strain factors {0.92, 1.08} → 2 strain probes per (phase,P)
For 1-2 representative phases at hottest T in regime:
  NVT MD with foundation, 5 snapshots → hot probes
```

DFT-label all in parallel with bootstrap pool labeling.

End-of-run check:

```
Tier-2 thresholds (defaults):
  E_MAE < 10 meV/at  on full set
  F_MAE < 100 meV/Å  on hot subset
  S_MAE < 10 meV/Å³  on strain subset
```

User overrides via `validation.thresholds:` block.

### 5.3 Tier 3 — full broad-indtest (opt-in, \~150 DFT calls)

`validation.tier: 3` enables:

- Hot MD probes at every T in regime (not just hottest).
- 5 strain factors instead of 2.
- Transition-region probes if PALLAS pathway provided.
- Full phonon dispersion check on at least one phase.

Costs more DFT, but generates publication-quality validation.

### 5.4 Supported-regime concept

Every report ends with:

```
Supported regime: P ∈ [0, 100] GPa, T ∈ [300, 3000] K, phases {dia, gra, hd, ...}
Out-of-regime use is NOT VALIDATED.
```

The grid IS the support claim. Honest, useful, prevents users from
extrapolating into regions where the model was never tested.

---

## 6. Repository layout

```
~/dev/Anvil/
├── DESIGN.md                # this doc
├── README.md                # quick start
├── pyproject.toml
├── anvil/
│   ├── __init__.py
│   ├── cli.py
│   ├── orchestrator.py
│   ├── config.py            # yaml loader + schema validation
│   ├── state.py             # state machine + checkpointing
│   ├── dft/
│   │   ├── vasp.py
│   │   ├── runner.py
│   │   └── validate.py
│   ├── ml/
│   │   ├── foundation.py
│   │   ├── ensemble.py
│   │   ├── trainer.py
│   │   └── compile.py
│   ├── al/
│   │   ├── generation/
│   │   │   ├── __init__.py
│   │   │   ├── local_entropy.py     # reformpy.EntropyMaximizingCalculator wrapper
│   │   │   ├── global_entropy.py    # ported from mlip-active-learn
│   │   │   ├── md_stratified.py     # per-(phase, P, T, k, mode) orchestrator
│   │   │   ├── reform_relax.py      # NEW: FP-targeted reformpy relax (CRISP-style)
│   │   │   └── anchor.py            # foundation MLIP only, regime-aligned
│   │   ├── acquisition.py           # QuotaMergeAcquisition (+ ablation variants)
│   │   └── stop.py
│   ├── validation/
│   │   ├── tier1_physics.py
│   │   ├── tier2_indtest.py
│   │   ├── tier3_full.py
│   │   └── thresholds.py
│   ├── hpc/
│   │   ├── slurm.py
│   │   ├── env.py
│   │   └── clusters/
│   │       ├── amareln.py
│   │       └── amarel3.py
│   └── report/
│       ├── html.py
│       └── plots.py
├── configs/
│   ├── defaults.yaml
│   ├── functionals/
│   │   ├── pbe.yaml
│   │   ├── r2scan_rvv10.yaml
│   │   └── scan.yaml
│   └── examples/
│       ├── carbon.yaml
│       ├── si.yaml
│       └── nacl.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    ├── architecture.md
    ├── acquisition_methods.md
    ├── validation.md
    └── troubleshooting.md
```

---

## 7. Dependencies

### 7.1 Hard dependencies (pyproject.toml)

```toml
[project]
name = "anvil-mlip"
requires-python = ">=3.10"
dependencies = [
    "ase >= 3.22",
    "numpy",
    "scipy",
    "matplotlib",
    "pyyaml",
    "click >= 8",
    "jinja2",
    "torch >= 2.0",
    "nequip >= 0.16",
    "reformpy >= 2.0",            # PI's group package; ZRG GitHub
    "libfp >= 3.1.2",             # via reformpy
    "mp-api >= 0.42",             # MP seed pulling
    "pymatgen",
]
```

### 7.2 External services

- **MP API**: optional, for `seeds.from_mp:` blocks. User-provided API key
  via env var `MP_API_KEY`.
- **Foundation model file**: `/scratch/lz432/allegro_finetune/allegro-oam-l-foundation.nequip.pth`
  on amareln/amarel3 (cluster registry resolves path).

### 7.3 HPC modules

- `intel/17.0.4` for VASP
- conda env `nequip` (PI's existing env)
- VASP binary path from cluster registry (`/home/lz432/apps/vasp.6.4.2/bin/vasp_std`)

---

## 8. v1 milestones

| Week | Deliverable                                                                                                                    | Validation                                       |
| ---- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| 1    | Skeleton + state machine + cluster registry. Pool A (port global\_entropy.py + md\_stratified.py) end-to-end *without* AL loop | Carbon smoke test on amareln, \~1 hour           |
| 2    | Pool B (reform\_relax.py, target mode (a)) + Pool C (anchor.py); carbon Round-0 bootstrap with all three pools                 | Inspect pool composition, FP-distance overlap    |
| 3    | AL loop + retrain; carbon ablation runs (A / A+B / A+C / A+B+C, 800 DFT each)                                                  | Compare against this paper's N=800 broad-indtest |
| 4    | Si + NaCl ablation; HTML report; CLI polish; example configs                                                                   | Three-system ablation table for paper            |
| 5    | Method paper draft                                                                                                             | —                                                |

The week-2 inspection step is important: we want to see that Pool B
candidates land in low-F bins and Pool C candidates land at near-zero F
*before* spending DFT on retraining. If the pool composition is off, no
amount of training will close the indtest gap.

---

## 9. Decisions logged

All design decisions for v1 confirmed. No remaining open questions before
week-1 code begins.

### PI-confirmed (2026-05-09, first batch)

1. **Repo location**: new repo at `/Users/li/dev/Anvil/`.
2. **reformpy dependency**: yes, hard dependency for local entropy +
   `Reform_Calculator`. We do NOT vendor reformpy; we pip-install it from
   the ZRG GitHub.
3. **Clusters**: amareln + amarel3 only for v1.
4. **Foundation model**: Allegro-OAM-L only for v1 (path
   `/scratch/lz432/allegro_finetune/allegro-oam-l-foundation.nequip.pth`).
5. **DFT code**: VASP only for v1.
6. **Naming**: Anvil. PyPI name TBD (`anvil-mlip` if `anvil` is taken).
7. **Global-entropy port**: direct copy of
   `/Users/li/dev/RA/mlip-active-learn/global_entropy_calculator.py`
   (313 LOC, no MPI, ASE-conformant) into
   `anvil/al/generation/global_entropy.py`. Stratification logic from
   `generate_structures.py` ported into `anvil/al/generation/md_stratified.py`.
8. **Reform-relax target FP**: default mode (a) — nearest seed-phase FP at
   the candidate's current pressure, drives toward closest known phase for
   high coverage. Opt-in mode (b) — farthest seed-phase FP, drives toward
   novel basins for phase discovery. Mode selected via yaml
   `reform_relax.mode: stability | discovery`.

### PI-confirmed (2026-05-09, second batch)

9. **Foundation model on amarel3**: Anvil ships a one-time
   `bootstrap_foundation` step that scp's the model from amareln to amarel3
   if missing. Idempotent — skips when content hash matches. Source of
   truth: amareln path `/scratch/lz432/allegro_finetune/allegro-oam-l-foundation.nequip.pth`.
10. **MP API key**: env var `MP_API_KEY`, documented in README. No
    `~/.anvil/credentials` fallback in v1 (avoid credential-file accidents).
11. **PALLAS integration**: v1 hook only — `reform_relax.py` accepts an
    optional `pallas_pathway_xyz` field in the yaml as a third
    initial-structure source. \~50 LOC. Full PALLAS pipeline (auto-discovery
    of pathways from seed phases) deferred to v2.
12. **Run directory layout**: `${SCRATCH}/anvil_runs/<system>/<run_id>/` on
    HPC scratch (avoids `$HOME` quotas; large pool xyzs and model
    checkpoints belong on scratch). A `.anvil/<system>` symlink in `cwd`
    points at the run dir so `anvil status` and `anvil report` resolve from
    the user's working directory without `--run-id` flags.
13. **License**: MIT (matches reformpy and libfp).
14. **Method-paper venue**: **npj Computational Materials**. Affects figure
    style — clean colorblind-safe palettes, larger font sizes, fewer panels
    per figure. Plot style guidelines folded into `anvil.report.plots`
    defaults. Submission target: end of week 5.

---

## 10. What we explicitly carry over from this paper's experience

### Operational (HPC + pipeline robustness)

- **Stress sign convention tags** in metadata (we lost a week on carbon).
- **Pool provenance hashes** (we lost a day on carbon n0800 corruption).
- **Shadow-variable conda activation** (we lost half a day on `$SIZE`).
- **Throttled SLURM submitter** (we lost hours to silent queue-cap truncation).
- **Login-node compute prohibition** (PI policy violation never recurs).
- **Broad-indtest construction logic** lifted into `validation/tier2_indtest.py`.
- **DFT convergence validation** before adding to training pool (NaCl B2 hot-MD
  timeout taught us: never silently accept failed VASP).

### Methodological (informs Anvil pool design directly)

- **Carbon F-gap on broad-indtest strain probes** (entropy 96 vs random 51
  meV/Å on the 50 strain probes; entropy 125 vs random 103 on hot MD).
  The strain-probe gap is the dominant signal — it's what Pool C anchor +
  Pool B reform-relax are designed to fix.
- **NaCl entropy E wins consistently** (4.6× on indtest, 5.9× on legacy).
  Pool A must remain dominant; Pool B + C should not dilute this win.
- **Si entropy F is competitive on indtest** (entropy 34 vs random 33 meV/Å,
  tied). Si is a baseline check that Pool B + C don't break a system that
  already works.
- **Random degrades with N on NaCl** (2.43 → 3.19 meV E\_MAE as N grows).
  This pattern is the cleanest evidence the entropy approach is necessary;
  the ablation table needs to preserve and ideally strengthen it.

### Data assets we re-use

- `/Users/li/dev/RA/mlip-active-learn/` for global-entropy port + `generate_structures.py`
  stratification logic.
- `docs/results_*_broad_*.npz` (broad-indtest evaluations) as the held-out
  test reference for the Anvil method paper.
- `docs/results_*_multiseed.npz` (legacy + indtest) as the Pool-A-only
  baseline column of the ablation table.

---

*End of v1 design. Revise here before any code is written.*
