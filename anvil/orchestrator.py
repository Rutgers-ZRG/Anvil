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
                              max_concurrent: int = 400, do_submit: bool = True) -> None:
        """BOOTSTRAP_SEEDS → BOOTSTRAP_DFT_QUEUED.

        Generate Pool C anchor candidates + Tier-2 validation seeds, write
        VASP inputs, optionally submit Slurm array.

        v1 piece-3 minimum: Pool C is strain-probe-only (no foundation MLIP
        relax). Pool A entropy + Pool B reform-relax come in week 2.
        """
        import json as _json

        from anvil.al.generation.anchor import AnchorConfig, generate_pool_c
        from anvil.dft.runner import submit_vasp_array
        from anvil.dft.vasp import (
            VASPInputs,
            load_functional,
            make_vasp_inputs,
            write_vasp_dir,
        )
        from anvil.hpc.clusters import get_cluster
        from anvil.validation.tier2_indtest import build_tier2_indtest_seeds

        cluster_cfg = get_cluster(self.cluster_name)

        # Resolve seed phases
        seed_phases_path = self.run_dir / "seed_phases.json"
        seed_phases = _json.loads(seed_phases_path.read_text())

        # Pool C anchor candidates
        anchor_cfg = AnchorConfig(
            phases=seed_phases,
            pressures_gpa=self.config.regime.pressures_gpa,
            temperatures_k=self.config.regime.temperatures_k,
            strain_factors=self.config.anchor.strain_factors,
        )
        pool_c = generate_pool_c(anchor_cfg)

        # Tier-2 validation seeds
        val_seeds = build_tier2_indtest_seeds(
            seed_phases=seed_phases,
            pressures_gpa=self.config.regime.pressures_gpa,
            temperatures_k=self.config.regime.temperatures_k,
        )

        # Write VASP inputs
        functional = load_functional(self.config.dft.functional)
        all_dirs: list[Path] = []
        for label, atoms_list in (("pool_c", pool_c), ("validation", val_seeds)):
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
                    pool=atoms.info.get("pool", ""),
                )
                sdir = label_dir / f"struct_{i:04d}"
                # POTCAR concatenation requires actual POTCAR files. Skip on local
                # development; cluster jobs will write POTCAR via cat_potcar=True.
                cat = do_submit and ssh_host is None
                write_vasp_dir(inp, sdir, cat_potcar=cat)
                all_dirs.append(sdir)

        # Persist struct dir manifest
        (self.run_dir / "bootstrap_struct_dirs.txt").write_text(
            "\n".join(str(d) for d in all_dirs) + "\n"
        )

        # Submit Slurm array (skip for local-only dry runs)
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

    def bootstrap(self, *, ssh_host: Optional[str] = None, do_submit: bool = True) -> None:
        """Run round-0 bootstrap up through BOOTSTRAP_DFT_QUEUED."""
        if self.ckpt.state == AnvilState.INIT:
            self.bootstrap_seeds()
        if self.ckpt.state == AnvilState.BOOTSTRAP_SEEDS:
            self.bootstrap_dft_queued(ssh_host=ssh_host, do_submit=do_submit)
