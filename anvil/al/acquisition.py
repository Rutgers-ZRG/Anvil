"""Acquisition: quota-merge of pools A/B/C with FP-distance dedupe.

See DESIGN.md §4.7.

v1 default: QuotaMergeAcquisition. No σ_F-based selection (deferred to v2).

Ablation variants for the method paper (§3.6):
  - PoolAOnlyAcquisition   — baseline (this paper's method)
  - PoolABAcquisition      — A + reform-relax only
  - PoolACAcquisition      — A + anchor only
  - QuotaMergeAcquisition  — A + B + C (default, full Anvil)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ase import Atoms


class Acquisition(Protocol):
    """Acquisition function protocol — pluggable for ablation."""

    def select(
        self,
        pool_a: list[Atoms],
        pool_b: list[Atoms],
        pool_c: list[Atoms],
        current_training_pool: list[Atoms],
    ) -> list[Atoms]:
        ...


@dataclass
class QuotaMergeAcquisition:
    """Default v1 acquisition: simple quota merge with FP-distance dedupe."""

    quotas: dict[str, int] = field(
        default_factory=lambda: {"A": 50, "B": 20, "C": 10}
    )
    fp_dedupe_threshold: float = 0.01    # distance below which two structures
                                          # are considered duplicates

    def select(
        self,
        pool_a: list[Atoms],
        pool_b: list[Atoms],
        pool_c: list[Atoms],
        current_training_pool: list[Atoms],
    ) -> list[Atoms]:
        """Merge with intra-pool ranking + cross-pool dedupe + final pool prune.

        1. Pool A ranked by entropy_gain (descending).
        2. Pool B ranked by relax-endpoint stability (low-F first).
        3. Pool C ranked by regime-coverage uniformity.
        4. Top-quota[X] from each pool.
        5. FP-distance dedupe across pool boundaries (drop later pool's dupe).
        6. Final FP-distance prune vs current_training_pool.
        7. Return ≤ sum(quotas) atoms.
        """
        raise NotImplementedError("week 2: see DESIGN.md §4.7")


@dataclass
class PoolAOnlyAcquisition:
    """Baseline ablation row: only Pool A, like the existing entropy-max paper."""

    n_select: int

    def select(
        self,
        pool_a: list[Atoms],
        pool_b: list[Atoms],
        pool_c: list[Atoms],
        current_training_pool: list[Atoms],
    ) -> list[Atoms]:
        raise NotImplementedError("week 3: ablation runs")


@dataclass
class PoolABAcquisition:
    """Ablation: A + reform-relax."""

    quotas: dict[str, int] = field(default_factory=lambda: {"A": 60, "B": 20})

    def select(
        self,
        pool_a: list[Atoms],
        pool_b: list[Atoms],
        pool_c: list[Atoms],
        current_training_pool: list[Atoms],
    ) -> list[Atoms]:
        raise NotImplementedError("week 3")


@dataclass
class PoolACAcquisition:
    """Ablation: A + anchor."""

    quotas: dict[str, int] = field(default_factory=lambda: {"A": 60, "C": 20})

    def select(
        self,
        pool_a: list[Atoms],
        pool_b: list[Atoms],
        pool_c: list[Atoms],
        current_training_pool: list[Atoms],
    ) -> list[Atoms]:
        raise NotImplementedError("week 3")
