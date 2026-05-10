"""Foundation MLIP registry. v1 supports Allegro-OAM-L only.

See DESIGN.md §4.4, §9 decisions #4 + #9.

Provides:
  - FoundationConfig: per-foundation metadata (paths, supported elements).
  - FoundationRegistry: lookup + ASE Calculator factory.
  - bootstrap_foundation(): idempotent scp helper to copy the model from
    amareln to amarel3 (or vice versa) when missing locally.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ase.calculators.calculator import Calculator


_DEFAULT_PATH = "/scratch/lz432/allegro_finetune/allegro-oam-l-foundation.nequip.pth"


@dataclass(frozen=True)
class FoundationConfig:
    name: str
    checkpoint_paths: dict[str, str]              # cluster_name -> abs path
    supported_elements: frozenset[int] = field(
        default_factory=frozenset
    )

    def path_for(self, cluster: str) -> str:
        if cluster not in self.checkpoint_paths:
            raise ValueError(
                f"Foundation {self.name!r} has no checkpoint registered "
                f"for cluster {cluster!r}. Known: {list(self.checkpoint_paths)}"
            )
        return self.checkpoint_paths[cluster]


class FoundationError(RuntimeError):
    """Raised when the foundation model cannot be loaded or located."""


class FoundationRegistry:
    """Static registry of foundation MLIPs available to Anvil."""

    _registry: dict[str, FoundationConfig] = {}

    @classmethod
    def register(cls, cfg: FoundationConfig) -> None:
        cls._registry[cfg.name] = cfg

    @classmethod
    def get(cls, name: str) -> FoundationConfig:
        if name not in cls._registry:
            raise FoundationError(
                f"Unknown foundation {name!r}. Registered: {list(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def make_calc(
        cls,
        name: str,
        *,
        cluster: str = "amareln",
        device: str = "cuda",
        elements: Optional[list[str]] = None,
    ) -> Calculator:
        """Load the foundation MLIP and return an ASE Calculator.

        If `elements` is provided, validates that the foundation covers all
        of them before loading (cheap pre-flight). Loads via
        `nequip.ase.NequIPCalculator.from_compiled_model`.
        """
        cfg = cls.get(name)
        path = cfg.path_for(cluster)
        if not Path(path).exists():
            raise FoundationError(
                f"Foundation checkpoint not found at {path}. "
                f"Run anvil.ml.foundation.bootstrap_foundation('{cluster}') to copy "
                f"from another registered cluster."
            )
        if elements is not None and cfg.supported_elements:
            from ase.data import atomic_numbers
            requested = {atomic_numbers[el] for el in elements}
            missing = requested - cfg.supported_elements
            if missing:
                from ase.data import chemical_symbols
                miss_syms = sorted(chemical_symbols[z] for z in missing)
                raise FoundationError(
                    f"Foundation {name!r} does not cover elements {miss_syms}."
                )

        try:
            from nequip.ase import NequIPCalculator
        except ImportError as exc:
            raise FoundationError(
                "nequip is required to load Allegro/NequIP foundations. "
                "Install nequip>=0.16."
            ) from exc

        return NequIPCalculator.from_compiled_model(path, device=device)


# ---------------------------------------------------------------------------
# Pre-registered foundations
# ---------------------------------------------------------------------------

ALLEGRO_OAM_L = FoundationConfig(
    name="allegro-oam-l-foundation",
    checkpoint_paths={
        "amareln": _DEFAULT_PATH,
        "amarel3": _DEFAULT_PATH,
    },
    # Element coverage will be filled in once we inspect the model metadata
    # (next step). For now, leave empty — the make_calc element-coverage check
    # is a no-op on an empty set.
    supported_elements=frozenset(),
)
FoundationRegistry.register(ALLEGRO_OAM_L)


# ---------------------------------------------------------------------------
# Cross-cluster bootstrap (one-time copy)
# ---------------------------------------------------------------------------


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ssh_alias(cluster: str) -> str:
    """Map cluster name to SSH host alias from ~/.ssh/config."""
    return {"amareln": "an", "amarel3": "amarel3"}[cluster]


def _remote_sha256(cluster: str, path: str) -> str | None:
    alias = _ssh_alias(cluster)
    cmd = ["ssh", "-o", "BatchMode=yes", alias,
           f"sha256sum {path} 2>/dev/null | awk '{{print $1}}'"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return None
    return out or None


def bootstrap_foundation(
    target_cluster: str,
    name: str = "allegro-oam-l-foundation",
    *,
    source_cluster: str = "amareln",
    via_local: Optional[str | Path] = None,
) -> str:
    """Idempotent scp of the foundation model from `source_cluster` to
    `target_cluster` if missing or hash-mismatched.

    Used during anvil submit when the run targets a cluster that doesn't have
    the model yet. via_local may be set to a Mac path to use as a relay (the
    common Mac-as-control pattern); if None, attempts direct scp source→target.

    Returns the target path.
    """
    cfg = FoundationRegistry.get(name)
    src_path = cfg.path_for(source_cluster)
    dst_path = cfg.path_for(target_cluster)

    src_hash = _remote_sha256(source_cluster, src_path)
    if src_hash is None:
        raise FoundationError(
            f"Cannot read foundation hash from {source_cluster}:{src_path}"
        )
    dst_hash = _remote_sha256(target_cluster, dst_path)
    if dst_hash == src_hash:
        return dst_path  # already in place

    src_alias = _ssh_alias(source_cluster)
    dst_alias = _ssh_alias(target_cluster)

    # Ensure target dir exists
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", dst_alias,
         f"mkdir -p {os.path.dirname(dst_path)}"],
        check=True,
    )

    if via_local is not None:
        local_tmp = Path(via_local)
        local_tmp.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["scp", "-o", "BatchMode=yes",
             f"{src_alias}:{src_path}", str(local_tmp)],
            check=True,
        )
        try:
            subprocess.run(
                ["scp", "-o", "BatchMode=yes",
                 str(local_tmp), f"{dst_alias}:{dst_path}"],
                check=True,
            )
        finally:
            try:
                local_tmp.unlink(missing_ok=True)
            except TypeError:                       # py < 3.8
                if local_tmp.exists():
                    local_tmp.unlink()
    else:
        # Direct scp from source to target (requires SSH between them)
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", src_alias,
             f"scp -o BatchMode=yes {src_path} {dst_alias}:{dst_path}"],
            check=True,
        )

    new_hash = _remote_sha256(target_cluster, dst_path)
    if new_hash != src_hash:
        raise FoundationError(
            f"Hash mismatch after bootstrap: src={src_hash[:8]}..., "
            f"dst={new_hash[:8] if new_hash else 'MISSING'}..."
        )
    return dst_path
