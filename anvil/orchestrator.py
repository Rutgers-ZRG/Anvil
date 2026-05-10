"""Main state-machine orchestrator. Drives the closed-loop AL.

See DESIGN.md §3.5 (algorithm) and §4.2 (state machine).

v1 piece-3 minimum: implements states INIT → BOOTSTRAP_SEEDS →
BOOTSTRAP_DFT_QUEUED. Subsequent states are stubbed for week 2.
"""

from __future__ import annotations

import datetime as _dt
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from anvil.config import AnvilConfig, load_config, to_canonical_json
from anvil.state import (
    AnvilState,
    StateCheckpoint,
    hash_bytes,
    load_checkpoint,
    save_checkpoint,
    transition,
)


def allocate_run_id(system: str) -> str:
    """`<UTC-yyyymmdd-HHMMSS>-<system>-<4-hex>`."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{system}-{secrets.token_hex(2)}"


def resolve_run_dir(system: str, run_id: str, *, scratch_root: str | Path) -> Path:
    """`${SCRATCH}/anvil_runs/<system>/<run_id>/`."""
    return Path(scratch_root) / "anvil_runs" / system / run_id


def make_cwd_symlink(run_dir: Path, system: str, *, cwd: Path | None = None) -> Path:
    """Create `.anvil/<system>` symlink in cwd pointing at run_dir.

    Idempotent: if symlink already points at run_dir, no-op. If it points
    elsewhere, replace.
    """
    cwd = cwd or Path.cwd()
    anvil_dir = cwd / ".anvil"
    anvil_dir.mkdir(exist_ok=True)
    link = anvil_dir / system
    target = run_dir.resolve()
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(os.readlink(link)) == target:
            return link
        link.unlink()
    link.symlink_to(target)
    return link


@dataclass
class Orchestrator:
    """Drives the state machine. v1 piece-3: round 0 only.

    Use:
        orch = Orchestrator.from_config_path("configs/examples/carbon.yaml",
                                             cluster="amareln")
        orch.bootstrap()              # INIT → BOOTSTRAP_DFT_QUEUED
    """

    config: AnvilConfig
    run_dir: Path
    cluster_name: str
    ckpt: StateCheckpoint

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        *,
        cluster: str = "amareln",
        scratch_root: str | Path | None = None,
        cwd: Path | None = None,
    ) -> "Orchestrator":
        cfg = load_config(config_path)
        run_id = allocate_run_id(cfg.system)
        from anvil.hpc.clusters import get_cluster
        cluster_cfg = get_cluster(cluster)
        scratch = Path(scratch_root) if scratch_root else Path(cluster_cfg.scratch_root)
        run_dir = resolve_run_dir(cfg.system, run_id, scratch_root=scratch)
        run_dir.mkdir(parents=True, exist_ok=True)
        make_cwd_symlink(run_dir, cfg.system, cwd=cwd)

        ckpt = StateCheckpoint(
            run_id=run_id,
            system=cfg.system,
            state=AnvilState.INIT,
            cluster=cluster,
            foundation_path=cluster_cfg.foundation_path,
            config_hash=hash_bytes(to_canonical_json(cfg).encode("utf-8")),
        )
        save_checkpoint(run_dir, ckpt)
        return cls(config=cfg, run_dir=run_dir, cluster_name=cluster, ckpt=ckpt)

    @classmethod
    def resume(cls, run_dir: str | Path, *, config_path: str | Path | None = None) -> "Orchestrator":
        """Resume from an existing checkpoint."""
        run_dir = Path(run_dir)
        ckpt = load_checkpoint(run_dir)
        # Re-load config: prefer the run-dir snapshot if present.
        snap = run_dir / "config.snapshot.yaml"
        if config_path:
            cfg = load_config(config_path)
        elif snap.exists():
            cfg = load_config(snap)
        else:
            raise RuntimeError(
                f"No config to resume from. Pass config_path or place a "
                f"config.snapshot.yaml at {run_dir}."
            )
        return cls(config=cfg, run_dir=run_dir, cluster_name=ckpt.cluster, ckpt=ckpt)

    def _save(self) -> None:
        save_checkpoint(self.run_dir, self.ckpt)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def bootstrap_seeds(self) -> None:
        """INIT → BOOTSTRAP_SEEDS.

        Resolve user seeds (from_mp or from_local) into POSCAR paths
        under <run_dir>/seeds/<phase>.vasp. Snapshot config to run_dir.
        """
        from shutil import copyfile

        seeds_dir = self.run_dir / "seeds"
        seeds_dir.mkdir(exist_ok=True)

        if self.config.seeds.from_local:
            phase_paths = {}
            for src in self.config.seeds.from_local:
                src_path = Path(src)
                if not src_path.is_absolute():
                    src_path = (Path.cwd() / src_path).resolve()
                if not src_path.exists():
                    raise FileNotFoundError(f"Seed POSCAR not found: {src_path}")
                # Phase name = filename stem (caller-controlled)
                phase = src_path.stem
                dst = seeds_dir / f"{phase}.vasp"
                copyfile(src_path, dst)
                phase_paths[phase] = str(dst)
            (self.run_dir / "seed_phases.json").write_text(
                __import__("json").dumps(phase_paths, indent=2)
            )
        elif self.config.seeds.from_mp:
            raise NotImplementedError(
                "MP seed pulling deferred to week 2 (requires MP_API_KEY env var)."
            )

        # Snapshot config so resume can re-read it
        snap = self.run_dir / "config.snapshot.yaml"
        if not snap.exists():
            # Re-emit yaml in canonical form
            import yaml
            from dataclasses import asdict as _asdict
            snap.write_text(yaml.safe_dump(_asdict(self.config), sort_keys=False))

        self.ckpt = transition(self.ckpt, AnvilState.BOOTSTRAP_SEEDS)
        self._save()

    def bootstrap_dft_queued(self, *, ssh_host: Optional[str] = None,
                              max_concurrent: int = 400, do_submit: bool = True,
                              foundation_calc: Optional["object"] = None,
                              skip_pools: tuple[str, ...] = ()) -> None:
        """BOOTSTRAP_SEEDS → BOOTSTRAP_DFT_QUEUED.

        Generate Pool A entropy MD + Pool B reform-relax + Pool C anchor +
        Tier-2 validation seeds, write VASP inputs, optionally submit Slurm
        array.

        Args:
            foundation_calc: Pre-loaded foundation MLIP. If None, loaded via
                FoundationRegistry from the registered cluster path. Pass an
                explicit calc for tests / local dev.
            skip_pools: Subset of {"A", "B", "C"} to skip. Used by the
                ablation runs (week 3): "A+B" → skip_pools=("C",), etc.
        """
        import json as _json

        from anvil.al.generation.anchor import AnchorConfig, generate_pool_c
        from anvil.al.generation.md_stratified import (
            StratifiedMDConfig, generate_pool_a,
        )
        from anvil.al.generation.reform_relax import (
            ReformRelaxConfig, generate_pool_b,
        )
        from anvil.dft.runner import submit_vasp_array
        from anvil.dft.vasp import (
            load_functional, make_vasp_inputs, write_vasp_dir,
        )
        from anvil.hpc.clusters import get_cluster
        from anvil.ml.foundation import FoundationRegistry
        from anvil.validation.tier2_indtest import build_tier2_indtest_seeds

        cluster_cfg = get_cluster(self.cluster_name)

        # Resolve seed phases
        seed_phases_path = self.run_dir / "seed_phases.json"
        seed_phases = _json.loads(seed_phases_path.read_text())

        # Build per-pool candidate lists
        pool_a: list = []
        pool_b: list = []
        pool_c: list = []

        # Pool A and B both need a foundation MLIP. Lazy-load only if needed.
        need_foundation = ("A" not in skip_pools) or ("B" not in skip_pools)
        if need_foundation and foundation_calc is None:
            foundation_calc = FoundationRegistry.make_calc(
                self.config.training.foundation_model,
                cluster=self.cluster_name,
                device="cuda",
                elements=self.config.elements,
            )

        # ---- Pool A — per-regime entropy MD ----
        if "A" not in skip_pools and self.config.generation_quotas.pool_a_entropy > 0:
            md_cfg = StratifiedMDConfig(
                phases=seed_phases,
                pressures_gpa=self.config.regime.pressures_gpa,
                temperatures_k=self.config.regime.temperatures_k,
                k_factors=self.config.entropy_md.k_factors,
                modes=self.config.entropy_md.modes,
                md_steps=self.config.entropy_md.md_steps,
                fp_cutoff=self.config.entropy_md.fp_cutoff,
                fp_natx=self.config.entropy_md.fp_natx,
                regularization=self.config.entropy_md.regularization,
                min_entropy_gain=self.config.entropy_md.min_entropy_gain,
            )
            fp_state = self.run_dir / "fp_dataset.npz"
            pool_a, _ = generate_pool_a(
                md_cfg, base_calc=foundation_calc,
                n_target=self.config.training.bootstrap_size,
                fp_dataset_path=fp_state,
                verbose=True,
            )

        # ---- Pool B — reform-relax ----
        if "B" not in skip_pools and self.config.generation_quotas.pool_b_reform > 0:
            rr_cfg = ReformRelaxConfig(
                mode=self.config.reform_relax.mode,
                perturbations_per_seed=self.config.reform_relax.pyxtal_n_per_round or 5,
                perturb_rattle=self.config.reform_relax.perturb_rattle,
                perturb_cell=self.config.reform_relax.perturb_cell,
                relax_max_steps=self.config.reform_relax.relax_max_steps,
                relax_fmax=self.config.reform_relax.relax_fmax,
                spglib_filter=self.config.reform_relax.spglib_filter,
            )
            pool_b = generate_pool_b(
                rr_cfg, base_calc=foundation_calc,
                seed_phases=seed_phases,
                n_target=self.config.generation_quotas.pool_b_reform,
                verbose=True,
            )

        # ---- Pool C — anchor (foundation-MLIP-free) ----
        if "C" not in skip_pools and self.config.generation_quotas.pool_c_anchor > 0:
            anchor_cfg = AnchorConfig(
                phases=seed_phases,
                pressures_gpa=self.config.regime.pressures_gpa,
                temperatures_k=self.config.regime.temperatures_k,
                strain_factors=self.config.anchor.strain_factors,
            )
            pool_c = generate_pool_c(anchor_cfg)

        # ---- Tier-2 validation seeds (always; foundation-free) ----
        val_seeds = build_tier2_indtest_seeds(
            seed_phases=seed_phases,
            pressures_gpa=self.config.regime.pressures_gpa,
            temperatures_k=self.config.regime.temperatures_k,
        )

        # Write VASP inputs
        functional = load_functional(self.config.dft.functional)
        all_dirs: list[Path] = []
        labelled = (
            ("pool_a", pool_a),
            ("pool_b", pool_b),
            ("pool_c", pool_c),
            ("validation", val_seeds),
        )
        for label, atoms_list in labelled:
            if not atoms_list:
                continue
            label_dir = self.run_dir / label
            label_dir.mkdir(exist_ok=True)
            for i, atoms in enumerate(atoms_list):
                inp = make_vasp_inputs(
                    atoms,
                    functional=self.config.dft.functional,
                    encut=self.config.dft.encut,
                    kspacing=self.config.dft.kspacing,
                    extra_incar=self.config.dft.extra_incar or None,
                    potcar_root=cluster_cfg.potcar_root,
                    sample_kind=atoms.info.get("sample_kind", ""),
                    pool=atoms.info.get("pool", label),
                )
                sdir = label_dir / f"struct_{i:04d}"
                cat = do_submit and ssh_host is None
                write_vasp_dir(inp, sdir, cat_potcar=cat)
                all_dirs.append(sdir)

        # Persist struct dir manifest grouped by pool for collect step
        manifest = {
            "pool_a": [str(d) for d in (self.run_dir / "pool_a").glob("struct_*")
                       if d.is_dir()] if pool_a else [],
            "pool_b": [str(d) for d in (self.run_dir / "pool_b").glob("struct_*")
                       if d.is_dir()] if pool_b else [],
            "pool_c": [str(d) for d in (self.run_dir / "pool_c").glob("struct_*")
                       if d.is_dir()] if pool_c else [],
            "validation": [str(d) for d in (self.run_dir / "validation").glob("struct_*")
                           if d.is_dir()] if val_seeds else [],
        }
        (self.run_dir / "bootstrap_manifest.json").write_text(
            _json.dumps(manifest, indent=2)
        )
        (self.run_dir / "bootstrap_struct_dirs.txt").write_text(
            "\n".join(str(d) for d in all_dirs) + "\n"
        )

        if do_submit:
            job_ids = submit_vasp_array(
                all_dirs,
                cluster=cluster_cfg,
                functional_yaml=functional,
                max_concurrent=max_concurrent,
                ssh_host=ssh_host,
            )
            self.ckpt.pending_job_ids = list(job_ids)

        self.ckpt.dft_calls_used += len(all_dirs)
        self.ckpt = transition(self.ckpt, AnvilState.BOOTSTRAP_DFT_QUEUED)
        self._save()

    def bootstrap_dft_done(self) -> None:
        """BOOTSTRAP_DFT_QUEUED → BOOTSTRAP_DFT_DONE.

        Collects converged VASP outputs from all four bootstrap dirs (pool_a,
        pool_b, pool_c, validation), drops unconverged + insane-force cells,
        and writes:
            <run_dir>/train.xyz   — pool_a + pool_b + pool_c (training data)
            <run_dir>/val.xyz     — validation set (Tier-2 indtest, never trained)

        Both xyz files carry pool / phase / sample_kind metadata.
        """
        import json as _json
        from anvil.dft.runner import collect_vasp_outputs
        from ase.io import write as ase_write

        manifest = _json.loads((self.run_dir / "bootstrap_manifest.json").read_text())

        train_atoms = []
        train_failed: list[str] = []
        for pool_label in ("pool_a", "pool_b", "pool_c"):
            dirs = manifest.get(pool_label, [])
            if not dirs:
                continue
            atoms_list, failed = collect_vasp_outputs(dirs)
            train_atoms.extend(atoms_list)
            train_failed.extend(failed)

        val_atoms, val_failed = collect_vasp_outputs(manifest.get("validation", []))

        train_xyz = self.run_dir / "train.xyz"
        val_xyz = self.run_dir / "val.xyz"
        if train_atoms:
            ase_write(str(train_xyz), train_atoms, format="extxyz")
        if val_atoms:
            ase_write(str(val_xyz), val_atoms, format="extxyz")

        report = {
            "train_collected": len(train_atoms),
            "train_failed": len(train_failed),
            "val_collected": len(val_atoms),
            "val_failed": len(val_failed),
            "train_xyz": str(train_xyz),
            "val_xyz": str(val_xyz),
            "failed_dirs": train_failed + val_failed,
        }
        (self.run_dir / "bootstrap_collect_report.json").write_text(
            _json.dumps(report, indent=2)
        )
        print(f"[bootstrap_dft_done] train: {len(train_atoms)} ok / "
              f"{len(train_failed)} fail; val: {len(val_atoms)} ok / "
              f"{len(val_failed)} fail")

        # Provenance
        from anvil.state import hash_file
        if train_xyz.exists():
            self.ckpt.pool_a_hash = hash_file(train_xyz)  # combined train hash
        if val_xyz.exists():
            self.ckpt.val_set_hash = hash_file(val_xyz)

        self.ckpt = transition(self.ckpt, AnvilState.BOOTSTRAP_DFT_DONE)
        self._save()

    def val_dft_done(self) -> None:
        """BOOTSTRAP_DFT_DONE → VAL_DFT_QUEUED → VAL_DFT_DONE.

        v1: validation set is labeled in the SAME bootstrap DFT array as the
        training pools (see bootstrap_dft_queued), so by the time we reach
        BOOTSTRAP_DFT_DONE the validation labels are already collected. This
        method just walks the state machine through the two val states.

        Future: when bootstrap is split (e.g. validation has its own DFT
        budget or runs on a different cluster), VAL_DFT_QUEUED becomes a
        real submit step.
        """
        self.ckpt = transition(self.ckpt, AnvilState.VAL_DFT_QUEUED)
        self._save()
        self.ckpt = transition(self.ckpt, AnvilState.VAL_DFT_DONE)
        self._save()

    def bootstrap_train(self, *, do_submit: bool = True,
                        ssh_host: Optional[str] = None,
                        n_models: Optional[int] = None,
                        seeds: Optional[list[int]] = None,
                        time: str = "12:00:00") -> None:
        """VAL_DFT_DONE → BOOTSTRAP_TRAIN.

        Train K=ensemble_size Allegro-OAM-L fine-tunes (different data.seeds)
        on the same train.xyz / val.xyz produced in bootstrap_dft_done.
        Submits Slurm jobs, persists job IDs in checkpoint.
        """
        from anvil.hpc.clusters import get_cluster
        from anvil.ml.trainer import train_ensemble

        cluster_cfg = get_cluster(self.cluster_name)
        train_xyz = self.run_dir / "train.xyz"
        val_xyz = self.run_dir / "val.xyz"
        if not train_xyz.exists():
            raise FileNotFoundError(
                f"train.xyz missing from {self.run_dir} — did "
                f"bootstrap_dft_done run successfully?"
            )

        ensemble_dir = self.run_dir / "ensemble_round0"
        result = train_ensemble(
            train_xyz=train_xyz,
            val_xyz=val_xyz,
            output_dir=ensemble_dir,
            cluster=cluster_cfg,
            n_models=n_models or self.config.training.ensemble_size,
            seeds=seeds,
            time=time,
            do_submit=do_submit,
            ssh_host=ssh_host,
        )

        # Persist for resume
        self.ckpt.pending_job_ids = list(result.job_ids)
        # Track ensemble run dirs separately (path strings — JSON-serializable)
        import json as _json
        (self.run_dir / "ensemble_round0_manifest.json").write_text(_json.dumps({
            "seeds": result.seeds,
            "run_dirs": [str(d) for d in result.run_dirs],
            "job_ids": result.job_ids,
            "template": result.template,
        }, indent=2))

        self.ckpt = transition(self.ckpt, AnvilState.BOOTSTRAP_TRAIN)
        self._save()

    def bootstrap_trained(self) -> None:
        """BOOTSTRAP_TRAIN → BOOTSTRAP_TRAINED.

        Collects compiled .nequip.pth checkpoints from each ensemble member.
        Caller is responsible for waiting until all training Slurm jobs have
        terminated before invoking this method.
        """
        import json as _json
        from anvil.ml.trainer import collect_compiled_models
        from anvil.state import hash_file

        manifest = _json.loads(
            (self.run_dir / "ensemble_round0_manifest.json").read_text()
        )
        run_dirs = [Path(d) for d in manifest["run_dirs"]]
        compiled = collect_compiled_models(run_dirs)

        if len(compiled) < len(run_dirs):
            print(f"[bootstrap_trained] WARNING: only {len(compiled)}/"
                  f"{len(run_dirs)} ensemble members produced compiled models. "
                  f"Run dirs without checkpoints: "
                  f"{[d.name for d in run_dirs if not any((d/'checkpoints'/n).exists() for n in ('best.nequip.pth','last.nequip.pth'))]}")

        # Persist hashes for provenance
        self.ckpt.ensemble_hashes = [hash_file(p) for p in compiled]
        # Also write the resolved paths for later retrieval
        (self.run_dir / "ensemble_round0_compiled.json").write_text(_json.dumps({
            "compiled_paths": [str(p) for p in compiled],
        }, indent=2))

        self.ckpt = transition(self.ckpt, AnvilState.BOOTSTRAP_TRAINED)
        self._save()

    def round_validated(
        self, *, device: str = "cuda",
    ) -> dict:
        """BOOTSTRAP_TRAINED → ROUND_CANDIDATES (after appending a monitor entry).

        Loads the K compiled ensemble members, evaluates Tier-2 indtest, writes
        a RoundMonitor row to the checkpoint, and prints the MAE summary.

        Note: per the state machine in DESIGN.md, after bootstrap_trained we
        enter the AL loop at ROUND_CANDIDATES. We use this method as the
        round-0 validation step (i.e. the trained ensemble is round 0; the
        loop's round 1 is the next AL pass). The transition target is
        ROUND_CANDIDATES so the loop machinery can pick up cleanly.
        """
        import json as _json
        from anvil.ml.ensemble import EnsembleCalculator
        from anvil.state import RoundMonitor
        from anvil.validation.tier2_indtest import evaluate_tier2

        compiled = _json.loads(
            (self.run_dir / "ensemble_round0_compiled.json").read_text()
        )["compiled_paths"]
        ensemble = EnsembleCalculator(compiled, device=device)
        val_xyz = self.run_dir / "val.xyz"

        metrics = evaluate_tier2(
            ensemble_calc=ensemble,
            indtest_xyz=val_xyz,
            thresholds=self.config.validation.thresholds,
        )

        # Persist + append monitor row
        (self.run_dir / "tier2_round0.json").write_text(
            _json.dumps(metrics, indent=2)
        )
        # Cumulative DFT count is already self.ckpt.dft_calls_used
        mon = RoundMonitor(
            round=0,
            pool_size=int(metrics["n_struct"]),
            dft_calls_used_cumulative=int(self.ckpt.dft_calls_used),
            val_e_mae=float(metrics["E_MAE"]),
            val_f_mae=float(metrics["F_MAE"]),
            val_s_mae=float(metrics["S_MAE"]),
        )
        self.ckpt.monitor.append(mon)

        # Stay on the validated state machine path: after BOOTSTRAP_TRAINED
        # we go to ROUND_CANDIDATES (round-1 of AL would now begin).
        self.ckpt = transition(self.ckpt, AnvilState.ROUND_CANDIDATES)
        self._save()

        print(f"[round-0 validation] E_MAE={metrics['E_MAE']:.2f} "
              f"F_MAE={metrics['F_MAE']:.1f} S_MAE={metrics['S_MAE']:.2f} "
              f"meV/{{at,Å,Å³}}; passed={metrics['passed']}")
        return metrics

    # ------------------------------------------------------------------
    # AL loop (round r ≥ 1)
    # ------------------------------------------------------------------

    def _ensemble_calc(self, device: str = "cuda"):
        """Build EnsembleCalculator from the most recent compiled-models manifest."""
        import json as _json
        from anvil.ml.ensemble import EnsembleCalculator
        # Pick the most recent ensemble_round{N}_compiled.json
        compiled_files = sorted(self.run_dir.glob("ensemble_round*_compiled.json"))
        if not compiled_files:
            raise RuntimeError("No trained ensemble found in run dir.")
        compiled = _json.loads(compiled_files[-1].read_text())["compiled_paths"]
        return EnsembleCalculator(compiled, device=device)

    def round_candidates(self, *, skip_pools: tuple[str, ...] = (),
                         device: str = "cuda") -> None:
        """ROUND_CANDIDATES: generate Pool A/B/C using current ensemble's mean as base."""
        import json as _json
        from anvil.al.generation.anchor import AnchorConfig, generate_pool_c
        from anvil.al.generation.md_stratified import (
            StratifiedMDConfig, generate_pool_a,
        )
        from anvil.al.generation.reform_relax import (
            ReformRelaxConfig, generate_pool_b,
        )

        seed_phases = _json.loads((self.run_dir / "seed_phases.json").read_text())
        ens = self._ensemble_calc(device=device)
        round_dir = self.run_dir / f"round_{self.ckpt.round + 1}"
        round_dir.mkdir(exist_ok=True)

        pool_a, pool_b, pool_c = [], [], []

        if "A" not in skip_pools and self.config.generation_quotas.pool_a_entropy > 0:
            md_cfg = StratifiedMDConfig(
                phases=seed_phases,
                pressures_gpa=self.config.regime.pressures_gpa,
                temperatures_k=self.config.regime.temperatures_k,
                k_factors=self.config.entropy_md.k_factors,
                modes=self.config.entropy_md.modes,
                md_steps=self.config.entropy_md.md_steps,
                fp_cutoff=self.config.entropy_md.fp_cutoff,
                fp_natx=self.config.entropy_md.fp_natx,
                regularization=self.config.entropy_md.regularization,
                min_entropy_gain=self.config.entropy_md.min_entropy_gain,
            )
            fp_state = self.run_dir / "fp_dataset.npz"
            pool_a, _ = generate_pool_a(
                md_cfg, base_calc=ens,
                n_target=self.config.generation_quotas.pool_a_entropy,
                fp_dataset_path=fp_state, verbose=True,
            )
        if "B" not in skip_pools and self.config.generation_quotas.pool_b_reform > 0:
            rr_cfg = ReformRelaxConfig(
                mode=self.config.reform_relax.mode,
                perturbations_per_seed=self.config.reform_relax.pyxtal_n_per_round or 5,
                perturb_rattle=self.config.reform_relax.perturb_rattle,
                perturb_cell=self.config.reform_relax.perturb_cell,
                relax_max_steps=self.config.reform_relax.relax_max_steps,
                relax_fmax=self.config.reform_relax.relax_fmax,
                spglib_filter=self.config.reform_relax.spglib_filter,
            )
            pool_b = generate_pool_b(
                rr_cfg, base_calc=ens, seed_phases=seed_phases,
                n_target=self.config.generation_quotas.pool_b_reform,
                verbose=True,
            )
        if "C" not in skip_pools and self.config.generation_quotas.pool_c_anchor > 0:
            anchor_cfg = AnchorConfig(
                phases=seed_phases,
                pressures_gpa=self.config.regime.pressures_gpa,
                temperatures_k=self.config.regime.temperatures_k,
                strain_factors=self.config.anchor.strain_factors,
            )
            pool_c = generate_pool_c(anchor_cfg)

        # Persist candidates per pool
        from ase.io import write as ase_write
        manifest = {}
        for label, atoms_list in (("pool_a", pool_a), ("pool_b", pool_b),
                                   ("pool_c", pool_c)):
            if not atoms_list:
                manifest[label] = []
                continue
            xyz = round_dir / f"{label}_candidates.xyz"
            ase_write(str(xyz), atoms_list, format="extxyz")
            manifest[label] = str(xyz)
        (round_dir / "candidates_manifest.json").write_text(
            _json.dumps(manifest, indent=2)
        )
        self.ckpt.round += 1
        self.ckpt = transition(self.ckpt, AnvilState.ROUND_CANDIDATES)
        self._save()

    def round_acquired(self) -> None:
        """ROUND_ACQUIRED: quota-merge candidates from the current round.

        v1: simple quota merge with no FP-dedupe (pools small enough that
        redundancy is unlikely on a single-system, ≤100 candidates/pool).
        Future: pluggable Acquisition class (PoolAOnly / PoolAB / etc.) and
        FP-distance dedupe.
        """
        import json as _json
        from ase.io import read as ase_read, write as ase_write

        round_dir = self.run_dir / f"round_{self.ckpt.round}"
        manifest = _json.loads(
            (round_dir / "candidates_manifest.json").read_text()
        )

        quotas = {
            "pool_a": self.config.generation_quotas.pool_a_entropy,
            "pool_b": self.config.generation_quotas.pool_b_reform,
            "pool_c": self.config.generation_quotas.pool_c_anchor,
        }
        selected = []
        for label, xyz in manifest.items():
            if not xyz:
                continue
            atoms_list = ase_read(xyz, index=":", format="extxyz")
            quota = quotas.get(label, len(atoms_list))
            atoms_list = atoms_list[:quota]
            for a in atoms_list:
                a.info.setdefault("pool", label[-1].upper())
            selected.extend(atoms_list)

        if selected:
            ase_write(str(round_dir / "selected.xyz"), selected,
                       format="extxyz")
        (round_dir / "selected_count.txt").write_text(str(len(selected)))

        self.ckpt = transition(self.ckpt, AnvilState.ROUND_ACQUIRED)
        self._save()

    def round_dft_queued(self, *, ssh_host: Optional[str] = None,
                         max_concurrent: int = 400) -> None:
        """ROUND_DFT_QUEUED: write VASP inputs + submit for the round's selected pool."""
        import json as _json
        from anvil.dft.runner import submit_vasp_array
        from anvil.dft.vasp import (
            load_functional, make_vasp_inputs, write_vasp_dir,
        )
        from anvil.hpc.clusters import get_cluster
        from ase.io import read as ase_read

        round_dir = self.run_dir / f"round_{self.ckpt.round}"
        selected_xyz = round_dir / "selected.xyz"
        if not selected_xyz.exists():
            raise FileNotFoundError(f"No selected.xyz at {selected_xyz}")
        atoms_list = ase_read(str(selected_xyz), index=":", format="extxyz")

        cluster_cfg = get_cluster(self.cluster_name)
        functional = load_functional(self.config.dft.functional)

        struct_dirs: list[Path] = []
        vasp_root = round_dir / "vasp_jobs"
        vasp_root.mkdir(exist_ok=True)
        for i, atoms in enumerate(atoms_list):
            inp = make_vasp_inputs(
                atoms,
                functional=self.config.dft.functional,
                encut=self.config.dft.encut,
                kspacing=self.config.dft.kspacing,
                extra_incar=self.config.dft.extra_incar or None,
                potcar_root=cluster_cfg.potcar_root,
                sample_kind=atoms.info.get("sample_kind", ""),
                pool=atoms.info.get("pool", ""),
            )
            sdir = vasp_root / f"struct_{i:04d}"
            write_vasp_dir(inp, sdir, cat_potcar=(ssh_host is None))
            struct_dirs.append(sdir)

        (round_dir / "struct_dirs.txt").write_text(
            "\n".join(str(d) for d in struct_dirs) + "\n"
        )

        job_ids = submit_vasp_array(
            struct_dirs, cluster=cluster_cfg, functional_yaml=functional,
            max_concurrent=max_concurrent, ssh_host=ssh_host,
        )
        self.ckpt.pending_job_ids = list(job_ids)
        self.ckpt.dft_calls_used += len(struct_dirs)
        self.ckpt = transition(self.ckpt, AnvilState.ROUND_DFT_QUEUED)
        self._save()

    def round_dft_done(self) -> None:
        """ROUND_DFT_DONE: collect VASP outputs + append to cumulative train.xyz."""
        import json as _json
        from anvil.dft.runner import collect_vasp_outputs
        from ase.io import read as ase_read, write as ase_write

        round_dir = self.run_dir / f"round_{self.ckpt.round}"
        struct_dirs = (round_dir / "struct_dirs.txt").read_text().split()

        new_atoms, failed = collect_vasp_outputs(struct_dirs)
        # Append to existing train.xyz (cumulative pool)
        train_xyz = self.run_dir / "train.xyz"
        if train_xyz.exists():
            existing = ase_read(str(train_xyz), index=":", format="extxyz")
        else:
            existing = []
        merged = existing + new_atoms
        ase_write(str(train_xyz), merged, format="extxyz")

        report = {
            "round": self.ckpt.round,
            "new_atoms": len(new_atoms),
            "failed": len(failed),
            "cumulative_train_size": len(merged),
            "failed_dirs": failed,
        }
        (round_dir / "collect_report.json").write_text(
            _json.dumps(report, indent=2)
        )
        print(f"[round {self.ckpt.round}] +{len(new_atoms)} new "
              f"({len(failed)} failed); train.xyz now {len(merged)} cells")

        from anvil.state import hash_file
        if train_xyz.exists():
            self.ckpt.pool_a_hash = hash_file(train_xyz)
        self.ckpt = transition(self.ckpt, AnvilState.ROUND_DFT_DONE)
        self._save()

    def round_train(self, *, do_submit: bool = True,
                    ssh_host: Optional[str] = None,
                    time: str = "12:00:00",
                    seeds: Optional[list[int]] = None) -> None:
        """ROUND_TRAIN: retrain ensemble on the cumulative training pool."""
        from anvil.hpc.clusters import get_cluster
        from anvil.ml.trainer import train_ensemble
        cluster_cfg = get_cluster(self.cluster_name)
        train_xyz = self.run_dir / "train.xyz"
        val_xyz = self.run_dir / "val.xyz"
        ensemble_dir = self.run_dir / f"ensemble_round{self.ckpt.round}"

        result = train_ensemble(
            train_xyz=train_xyz, val_xyz=val_xyz,
            output_dir=ensemble_dir, cluster=cluster_cfg,
            n_models=self.config.training.ensemble_size,
            seeds=seeds, time=time,
            do_submit=do_submit, ssh_host=ssh_host,
        )
        self.ckpt.pending_job_ids = list(result.job_ids)
        import json as _json
        (self.run_dir / f"ensemble_round{self.ckpt.round}_manifest.json").write_text(
            _json.dumps({
                "seeds": result.seeds,
                "run_dirs": [str(d) for d in result.run_dirs],
                "job_ids": result.job_ids,
                "template": result.template,
            }, indent=2)
        )
        self.ckpt = transition(self.ckpt, AnvilState.ROUND_TRAIN)
        self._save()

    def round_trained(self) -> None:
        """ROUND_TRAINED: collect compiled checkpoints for the new ensemble."""
        import json as _json
        from anvil.ml.trainer import collect_compiled_models
        from anvil.state import hash_file

        manifest = _json.loads(
            (self.run_dir / f"ensemble_round{self.ckpt.round}_manifest.json").read_text()
        )
        run_dirs = [Path(d) for d in manifest["run_dirs"]]
        compiled = collect_compiled_models(run_dirs)
        self.ckpt.ensemble_hashes = [hash_file(p) for p in compiled]
        (self.run_dir / f"ensemble_round{self.ckpt.round}_compiled.json").write_text(
            _json.dumps({"compiled_paths": [str(p) for p in compiled]}, indent=2)
        )
        self.ckpt = transition(self.ckpt, AnvilState.ROUND_TRAINED)
        self._save()

    def round_validated_and_decide(self, *, device: str = "cuda") -> dict:
        """ROUND_TRAINED → ROUND_VALIDATED → {ROUND_CANDIDATES | FINAL_COMPILE}.

        Eval Tier-2, append RoundMonitor, and decide whether to loop another
        round or finalize.
        """
        import json as _json
        from anvil.al.stop import StopCriteria, should_stop
        from anvil.state import RoundMonitor
        from anvil.validation.tier2_indtest import evaluate_tier2

        ens = self._ensemble_calc(device=device)
        val_xyz = self.run_dir / "val.xyz"
        metrics = evaluate_tier2(
            ens, val_xyz, thresholds=self.config.validation.thresholds,
        )
        (self.run_dir / f"tier2_round{self.ckpt.round}.json").write_text(
            _json.dumps(metrics, indent=2)
        )
        mon = RoundMonitor(
            round=int(self.ckpt.round),
            pool_size=int(metrics["n_struct"]),
            dft_calls_used_cumulative=int(self.ckpt.dft_calls_used),
            val_e_mae=float(metrics["E_MAE"]),
            val_f_mae=float(metrics["F_MAE"]),
            val_s_mae=float(metrics["S_MAE"]),
        )
        self.ckpt.monitor.append(mon)
        self.ckpt = transition(self.ckpt, AnvilState.ROUND_VALIDATED)
        self._save()

        # Decide: loop or finalize
        criteria = StopCriteria(
            budget_dft_calls=self.config.budget.max_dft_calls,
            max_walltime_h=self.config.budget.max_walltime_h,
        )
        stop, reason = should_stop(self.ckpt, criteria)
        print(f"[round {self.ckpt.round}] E={metrics['E_MAE']:.2f} "
              f"F={metrics['F_MAE']:.1f} S={metrics['S_MAE']:.2f} "
              f"meV/{{at,Å,Å³}}; passed={metrics['passed']}; "
              f"stop={stop}{f' ({reason})' if reason else ''}")
        if stop:
            self.ckpt = transition(self.ckpt, AnvilState.FINAL_COMPILE)
            self._save()
        else:
            self.ckpt = transition(self.ckpt, AnvilState.ROUND_CANDIDATES)
            self.ckpt.round += 1
            # Stay at ROUND_CANDIDATES — the loop driver continues
            self._save()
        return metrics

    def bootstrap(self, *, ssh_host: Optional[str] = None, do_submit: bool = True,
                  foundation_calc: Optional["object"] = None,
                  skip_pools: tuple[str, ...] = ()) -> None:
        """Run round-0 bootstrap up through BOOTSTRAP_DFT_QUEUED."""
        if self.ckpt.state == AnvilState.INIT:
            self.bootstrap_seeds()
        if self.ckpt.state == AnvilState.BOOTSTRAP_SEEDS:
            self.bootstrap_dft_queued(
                ssh_host=ssh_host, do_submit=do_submit,
                foundation_calc=foundation_calc, skip_pools=skip_pools,
            )
