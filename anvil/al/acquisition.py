"""Acquisition: quota-merge of pools A/B/C with FP-distance dedupe.

See DESIGN.md §4.7. Implementation authored by Nate (PR2 draft,
summer_research_2026/nate/anvil_patches/acquisition_impl.py), reviewed and
applied locally by mlip-trainer 2026-07-16 (Anvil has no remote yet — gate A1;
this lands as Nate's PR2 once the remote exists).

Design goals
------------
1. Fill in the four `select()` bodies against the ACTUAL metadata the Anvil
   generators stamp (verified by reading the generator modules):
     - Pool A (`md_stratified.generate_pool_a`): info["entropy_gain"] (float),
       info["phase"], info["pressure_gpa"], info["temperature_k"], info["k_factor"].
     - Pool B (`reform_relax.generate_pool_b`): info["reform_lambda"], info["reform_mode"].
       ⚠ NO stored force / stability metric — see `_rank_pool_b` note.
     - Pool C (`anchor.generate_pool_c`): info["phase"], info["pressure_gpa"],
       info["sample_kind"] (=strain factor).
2. Reuse the real FP machinery: `anvil.al.generation.global_entropy.compute_fingerprints`.
3. Never silently drop on a missing optional dep (libfp/reformpy) — if FP is
   unavailable, skip dedupe with a warning (matches Anvil's house rule, cf.
   reform_relax._has_finite_symmetry).

Ablation → variant map (used by the round_acquired rewrite's factory):
    skip B and C  -> PoolAOnlyAcquisition
    skip C only   -> PoolABAcquisition
    skip B only   -> PoolACAcquisition
    none skipped  -> QuotaMergeAcquisition
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
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


# ----------------------------------------------------------------------------
# FP helpers (shared)  — config-level fingerprint = mean over per-atom libfp fp.
# ----------------------------------------------------------------------------

def _config_fp(atoms: Atoms, cutoff: float, natx: int) -> Optional[np.ndarray]:
    """Return a (natx,) config descriptor, or None if FP backend is unavailable.

    Uses the same libfp path as GlobalEntropyCalculator. Import is done lazily
    and guarded so that a missing libfp/reformpy degrades to "no dedupe" rather
    than crashing acquisition.
    """
    try:
        from anvil.al.generation.global_entropy import compute_fingerprints
    except ImportError:
        return None
    try:
        fp = compute_fingerprints(atoms, cutoff, natx)["fp"]  # (nat, natx)
    except Exception:
        return None
    return np.asarray(fp, dtype=np.float64).mean(axis=0)      # (natx,)


def _fp_dedupe(
    candidates: list[Atoms],
    reference_fps: list[np.ndarray],
    threshold: float,
    cutoff: float,
    natx: int,
) -> list[Atoms]:
    """Keep candidates whose config-FP is > `threshold` from every reference FP
    AND from every earlier kept candidate. Reference FPs are extended in place
    with each kept candidate so intra-batch dupes are also removed.

    If the FP backend is unavailable, returns `candidates` unchanged with a
    warning (never silently drops).
    """
    kept: list[Atoms] = []
    backend_ok = True
    for a in candidates:
        fp = _config_fp(a, cutoff, natx)
        if fp is None:
            backend_ok = False
            kept.append(a)
            continue
        if all(np.linalg.norm(fp - r) > threshold for r in reference_fps):
            kept.append(a)
            reference_fps.append(fp)
    if not backend_ok:
        warnings.warn(
            "acquisition FP-dedupe skipped: libfp/reformpy unavailable in this "
            "env (see nate/SETUP.md gate A5). Candidates passed through un-deduped."
        )
    return kept


# ----------------------------------------------------------------------------
# Per-pool ranking
# ----------------------------------------------------------------------------

def _rank_pool_a(pool: list[Atoms]) -> list[Atoms]:
    """Rank Pool A by entropy_gain (descending), stable.

    NOTE: when the run bypasses the entropy filter (min_entropy_gain <= -1e6,
    as the carbon ablation does), every snapshot carries entropy_gain == 0.0, so
    this is a stable no-op and the upstream n_target subsample already chose the
    set. That is fine and intended for v1.
    """
    return sorted(pool, key=lambda a: -float(a.info.get("entropy_gain", 0.0)))


def _rank_pool_b(pool: list[Atoms]) -> list[Atoms]:
    """Rank Pool B by relax-endpoint stability (low max-force first).

    ⚠ v1 GAP: generate_pool_b (reform_relax.py) does NOT stamp the final max
    force into atoms.info — it only keeps info["reform_lambda"]/["reform_mode"].
    So there is no stored stability metric to rank on. Two honest options:
      (1) [preferred, small PR2 add] have generate_pool_b stamp
          info["max_force"] = float(np.abs(final.get_forces()).max()) right
          before `final.calc = None` (forces are already available from the
          relaxed calc). Then this ranker sorts on it.
      (2) fall back to input order (already relaxed, roughly ordered).
    This impl uses info["max_force"] if present, else input order — so it works
    today and improves the moment option (1) lands.
    """
    if any("max_force" in a.info for a in pool):
        return sorted(pool, key=lambda a: float(a.info.get("max_force", np.inf)))
    return list(pool)


def _rank_pool_c(pool: list[Atoms]) -> list[Atoms]:
    """Rank Pool C for regime-coverage uniformity via round-robin over bins.

    Bin key = (phase, pressure_gpa, sample_kind[=strain factor]). Emitting one
    item per bin in rounds means the top-quota slice spreads evenly across the
    regime grid instead of piling onto one (phase, P).
    """
    from collections import OrderedDict

    bins: "OrderedDict[tuple, list[Atoms]]" = OrderedDict()
    for a in pool:
        key = (a.info.get("phase"), a.info.get("pressure_gpa"),
               a.info.get("sample_kind"))
        bins.setdefault(key, []).append(a)
    ordered: list[Atoms] = []
    buckets = list(bins.values())
    while any(buckets):
        for b in buckets:
            if b:
                ordered.append(b.pop(0))
    return ordered


def _stamp_pool(atoms_list: list[Atoms], letter: str) -> None:
    for a in atoms_list:
        a.info.setdefault("pool", letter)


# ----------------------------------------------------------------------------
# Acquisition variants
# ----------------------------------------------------------------------------

@dataclass
class PoolAOnlyAcquisition:
    """Baseline ablation row: only Pool A (like the existing entropy-max paper)."""

    n_select: int
    fp_dedupe_threshold: float = 0.01
    fp_cutoff: float = 4.0
    fp_natx: int = 50

    def select(self, pool_a, pool_b, pool_c, current_training_pool):
        ranked = _rank_pool_a(pool_a)[: self.n_select]
        _stamp_pool(ranked, "A")
        ref = [fp for fp in
               (_config_fp(a, self.fp_cutoff, self.fp_natx)
                for a in current_training_pool) if fp is not None]
        return _fp_dedupe(ranked, ref, self.fp_dedupe_threshold,
                          self.fp_cutoff, self.fp_natx)


@dataclass
class _QuotaMergeBase:
    """Shared quota-merge logic for A(+B)(+C) variants."""

    quotas: dict[str, int] = field(default_factory=lambda: {"A": 50, "B": 20, "C": 10})
    fp_dedupe_threshold: float = 0.01
    fp_cutoff: float = 4.0
    fp_natx: int = 50

    def _merge(self, pool_a, pool_b, pool_c, current_training_pool):
        # 1. rank within pool, 2. take top-quota[X] from each
        picks: list[Atoms] = []
        if self.quotas.get("A", 0) > 0:
            a = _rank_pool_a(pool_a)[: self.quotas["A"]]
            _stamp_pool(a, "A")
            picks.append(a)
        if self.quotas.get("B", 0) > 0:
            b = _rank_pool_b(pool_b)[: self.quotas["B"]]
            _stamp_pool(b, "B")
            picks.append(b)
        if self.quotas.get("C", 0) > 0:
            c = _rank_pool_c(pool_c)[: self.quotas["C"]]
            _stamp_pool(c, "C")
            picks.append(c)

        # 3. FP-dedupe ACROSS pool boundaries (A kept first, later pools deduped
        #    against it), then 4. final prune vs current_training_pool.
        ref = [fp for fp in
               (_config_fp(a, self.fp_cutoff, self.fp_natx)
                for a in current_training_pool) if fp is not None]
        selected: list[Atoms] = []
        for pool in picks:                 # A, then B, then C (priority order)
            selected.extend(
                _fp_dedupe(pool, ref, self.fp_dedupe_threshold,
                           self.fp_cutoff, self.fp_natx)
            )
        return selected                    # 5. ≤ sum(quotas) atoms


@dataclass
class PoolABAcquisition(_QuotaMergeBase):
    """Ablation: A + reform-relax."""

    quotas: dict[str, int] = field(default_factory=lambda: {"A": 60, "B": 20})

    def select(self, pool_a, pool_b, pool_c, current_training_pool):
        return self._merge(pool_a, pool_b, [], current_training_pool)


@dataclass
class PoolACAcquisition(_QuotaMergeBase):
    """Ablation: A + anchor."""

    quotas: dict[str, int] = field(default_factory=lambda: {"A": 60, "C": 20})

    def select(self, pool_a, pool_b, pool_c, current_training_pool):
        return self._merge(pool_a, [], pool_c, current_training_pool)


@dataclass
class QuotaMergeAcquisition(_QuotaMergeBase):
    """Default v1 acquisition: quota merge of A+B+C with FP-distance dedupe.

    ⚠ Depends on libfp (or reformpy fallback) for the FP-dedupe steps (3–4). If
    that backend is absent the merge still runs but dedupe is skipped with a
    warning — so QuotaMerge is on Nate's libfp critical path (SETUP.md gate A5).
    """

    quotas: dict[str, int] = field(default_factory=lambda: {"A": 50, "B": 20, "C": 10})

    def select(self, pool_a, pool_b, pool_c, current_training_pool):
        return self._merge(pool_a, pool_b, pool_c, current_training_pool)


# ----------------------------------------------------------------------------
# Factory used by the round_acquired() rewrite
# ----------------------------------------------------------------------------

def build_acquisition(config) -> Acquisition:
    """Pick an Acquisition from an AnvilConfig's generation_quotas.

    A pool with quota 0 is excluded (this is exactly how the ablation encodes
    A_only / A+B / A+C — see scripts/ablation_carbon_n200.py::ABLATION_ROWS).
    """
    q = config.generation_quotas
    fp_kw = dict(
        fp_dedupe_threshold=0.01,
        fp_cutoff=float(getattr(config.entropy_md, "fp_cutoff", 4.0)),
        fp_natx=int(getattr(config.entropy_md, "fp_natx", 50)),
    )
    a, b, c = q.pool_a_entropy, q.pool_b_reform, q.pool_c_anchor
    if b == 0 and c == 0:
        return PoolAOnlyAcquisition(n_select=a, **fp_kw)
    if c == 0:
        return PoolABAcquisition(quotas={"A": a, "B": b}, **fp_kw)
    if b == 0:
        return PoolACAcquisition(quotas={"A": a, "C": c}, **fp_kw)
    return QuotaMergeAcquisition(quotas={"A": a, "B": b, "C": c}, **fp_kw)
