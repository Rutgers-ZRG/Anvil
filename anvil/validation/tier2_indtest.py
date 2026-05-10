"""Tier 2 — mini broad-indtest. Default validation, ~50 DFT calls.

See DESIGN.md §5.2.

Built once at bootstrap, never enters training. Construction logic ports
build_broad_indtest.py from mlip-active-learn but in v1 piece-3 minimum
generates ONLY strain probes (no hot-MD, since hot MD needs the foundation
MLIP wired). Strain factors are wider than Pool C anchor's so the validation
set is genuinely independent.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read as ase_read

from anvil.al.generation.anchor import _apply_isotropic_strain, _read_seed


# Wider strain factors than Pool C anchor — Tier-2 should probe a broader
# region than training data. Matches build_broad_indtest.py defaults.
TIER2_STRAIN_FACTORS: tuple[float, float] = (0.92, 1.08)


def build_tier2_indtest_seeds(
    seed_phases: dict[str, str | Path],
    pressures_gpa: list[float],
    temperatures_k: list[float] | None = None,
    *,
    strain_factors: tuple[float, float] = TIER2_STRAIN_FACTORS,
    n_hot_per_phase: int = 0,             # v1 piece-3: 0 (no foundation MLIP)
) -> list[Atoms]:
    """Construct the Tier-2 mini broad-indtest seeds.

    v1 piece-3 minimum: strain probes only, 2 factors × 5 P × N_phases.
    For carbon (5 phases × 5 P × 2 factors) = 50 cells. Matches DESIGN.md §5.2.

    n_hot_per_phase reserved for week-2 foundation MLIP integration.
    """
    out: list[Atoms] = []
    for phase, seed_path in seed_phases.items():
        seed = _read_seed(seed_path)
        for P in pressures_gpa:
            for factor in strain_factors:
                a = _apply_isotropic_strain(seed, factor)
                a.info["pool"] = "validation"
                a.info["phase"] = phase
                a.info["pressure_gpa"] = float(P)
                a.info["temperature_k"] = 0
                a.info["sample_kind"] = f"strain_{factor:.2f}"
                out.append(a)

    if n_hot_per_phase > 0:
        raise NotImplementedError(
            "Hot MD probes require foundation MLIP integration (week 2)."
        )
    return out


def evaluate_tier2(
    ensemble_calc: Calculator,
    indtest_xyz: str | Path,
    thresholds: Optional[dict] = None,
) -> dict:
    """Evaluate ensemble on a labeled Tier-2 set; return MAE + threshold check.

    Reuses the eval logic from the prior paper's eval_multiseed.py:
      - E_MAE per atom (mean over structures)
      - F_MAE flattened over (atoms × xyz components × structures)
      - S_MAE flattened over Voigt components × structures

    Returns dict with units **meV/{atom, Å, Å³}**:
        {
            "E_MAE": float,
            "F_MAE": float,
            "S_MAE": float,
            "subset": {<sample_kind>: {"E_MAE": ..., "F_MAE": ..., "S_MAE": ...}},
            "n_struct": int,
            "passed": bool,
            "failed_thresholds": list[str],
        }
    """
    from anvil.validation.thresholds import (
        DEFAULT_THRESHOLDS,
        check_thresholds as _check,
    )

    test_atoms = ase_read(str(indtest_xyz), index=":", format="extxyz")
    if not test_atoms:
        return {
            "E_MAE": float("nan"), "F_MAE": float("nan"), "S_MAE": float("nan"),
            "n_struct": 0, "passed": False,
            "failed_thresholds": ["empty_set"], "subset": {},
        }

    # References (from VASP labels)
    e_ref = np.array([a.get_potential_energy() / len(a) for a in test_atoms])
    f_ref = [a.get_forces() for a in test_atoms]
    s_ref = []
    for a in test_atoms:
        try:
            s_ref.append(np.array(a.get_stress()))
        except Exception:
            s_ref.append(np.full(6, np.nan))

    # Predictions
    e_pred = []
    f_pred = []
    s_pred = []
    for a in test_atoms:
        ac = a.copy()
        ac.calc = ensemble_calc
        e_pred.append(ac.get_potential_energy() / len(a))
        f_pred.append(np.array(ac.get_forces()))
        try:
            s_pred.append(np.array(ac.get_stress()))
        except Exception:
            s_pred.append(np.full(6, np.nan))
    e_pred = np.array(e_pred)

    e_mae_eV = float(np.mean(np.abs(e_pred - e_ref)))
    f_mae_eV = float(np.mean(np.concatenate(
        [np.abs(np.array(p) - np.array(r)).flatten()
         for p, r in zip(f_pred, f_ref)]
    )))
    s_mae_eVA3 = float(np.mean(np.concatenate(
        [np.abs(np.array(p) - np.array(r)).flatten()
         for p, r in zip(s_pred, s_ref)]
    )))

    # Convert to display units (meV/{atom,Å,Å³})
    metrics = {
        "E_MAE": e_mae_eV * 1000,
        "F_MAE": f_mae_eV * 1000,
        "S_MAE": s_mae_eVA3 * 1000,
        "n_struct": len(test_atoms),
    }

    # Subset metrics by sample_kind ("strain_0.92" vs "hot_snap_0", ...)
    subsets: dict[str, dict] = {}
    by_kind: dict[str, list[int]] = {}
    for i, a in enumerate(test_atoms):
        kind = a.info.get("sample_kind", "unknown")
        # Coalesce strain factors into one bucket; keep hot snaps named
        bucket = "strain" if kind.startswith("strain_") else \
                 ("hot" if kind.startswith("hot") else kind)
        by_kind.setdefault(bucket, []).append(i)
    for bucket, idx in by_kind.items():
        if not idx:
            continue
        e_b = float(np.mean(np.abs(e_pred[idx] - e_ref[idx]))) * 1000
        f_b = float(np.mean(np.concatenate(
            [np.abs(np.array(f_pred[i]) - np.array(f_ref[i])).flatten()
             for i in idx]
        ))) * 1000
        s_b = float(np.mean(np.concatenate(
            [np.abs(np.array(s_pred[i]) - np.array(s_ref[i])).flatten()
             for i in idx]
        ))) * 1000
        subsets[bucket] = {"E_MAE": e_b, "F_MAE": f_b, "S_MAE": s_b,
                            "n": len(idx)}
    metrics["subset"] = subsets

    passed, failed = _check(metrics, thresholds or DEFAULT_THRESHOLDS)
    metrics["passed"] = passed
    metrics["failed_thresholds"] = failed
    return metrics
