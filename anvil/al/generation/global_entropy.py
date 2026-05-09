"""Dataset-aware global entropy calculator (Subramanyam & Perez 2025 + this work).

PORTED from /Users/li/dev/RA/mlip-active-learn/global_entropy_calculator.py
(313 LOC, no MPI, ASE-conformant). See DESIGN.md §3.2, §9 decision #7.

Strategy: E = E_base - k · log det Σ(D ∪ current)
where Σ is the feature covariance over the entire dataset (existing + current).

Two modes:
  - per_atom:   covariance over individual atomic fingerprints  (disordered)
  - per_config: covariance over config-mean fingerprints        (ordered)

Public API (matches the original file for drop-in compatibility):
  - FingerprintDataset(fp_dim, reg=1e-3)
      .add(fps), .get_stats(), .get_stats_with(extra_fps),
      .log_det(extra_fps=None), .save(path), .load(path)
  - GlobalEntropyCalculator(calculator, dataset, k_factor=5.0, cutoff=5.0,
                            natx=50, mode='per_atom')
      .calculate(...), .add_config(atoms), .entropy_gain(atoms)
"""

from __future__ import annotations

# Week-1 plan: COPY the file verbatim from
#   /Users/li/dev/RA/mlip-active-learn/global_entropy_calculator.py
# into this module. Only adaptation: import libfp via reformpy fallback
# (reformpy.libfppy if libfp unavailable). The acquisition layer in
# anvil.al.acquisition reuses FingerprintDataset for FP-distance dedupe.

raise NotImplementedError(
    "week 1: direct port of global_entropy_calculator.py from mlip-active-learn"
)
