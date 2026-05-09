"""Unit tests for anchor + tier2 strain-probe generation (v1 piece-3 minimum)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write

from anvil.al.generation.anchor import (
    AnchorConfig,
    _apply_isotropic_strain,
    generate_pool_c,
)
from anvil.validation.tier2_indtest import build_tier2_indtest_seeds


def make_dummy_seed(tmp_path: Path, name: str) -> Path:
    """Tiny diamond-like cell."""
    a = 3.5
    atoms = Atoms(
        symbols=["Si", "Si"],
        positions=[(0, 0, 0), (a / 2, a / 2, a / 2)],
        cell=[(a, 0, 0), (0, a, 0), (0, 0, a)],
        pbc=True,
    )
    p = tmp_path / f"{name}.vasp"
    ase_write(str(p), atoms, format="vasp", vasp5=True, direct=True)
    return p


def test_isotropic_strain_scales_volume() -> None:
    a = Atoms("Si", positions=[(0, 0, 0)], cell=[(3, 0, 0), (0, 3, 0), (0, 0, 3)], pbc=True)
    s = _apply_isotropic_strain(a, 1.10)
    assert np.isclose(s.get_volume(), a.get_volume() * 1.10**3)
    s2 = _apply_isotropic_strain(a, 0.92)
    assert np.isclose(s2.get_volume(), a.get_volume() * 0.92**3)


def test_pool_c_count(tmp_path: Path) -> None:
    p1 = make_dummy_seed(tmp_path, "phase_a")
    p2 = make_dummy_seed(tmp_path, "phase_b")
    cfg = AnchorConfig(
        phases={"phase_a": str(p1), "phase_b": str(p2)},
        pressures_gpa=[0, 10, 30],
        strain_factors=[0.95, 1.05],
    )
    pool = generate_pool_c(cfg)
    assert len(pool) == 2 * 3 * 2  # 2 phases × 3 P × 2 strain
    # Metadata sanity
    assert all(a.info["pool"] == "C" for a in pool)
    kinds = {a.info["sample_kind"] for a in pool}
    assert kinds == {"strain_0.95", "strain_1.05"}
    # Distinct phases distributed
    phases = {a.info["phase"] for a in pool}
    assert phases == {"phase_a", "phase_b"}


def test_tier2_count_and_factors(tmp_path: Path) -> None:
    p1 = make_dummy_seed(tmp_path, "diamond")
    p2 = make_dummy_seed(tmp_path, "graphite")
    seeds = build_tier2_indtest_seeds(
        seed_phases={"diamond": str(p1), "graphite": str(p2)},
        pressures_gpa=[0, 30, 100],
    )
    # 2 phases × 3 P × 2 factors = 12
    assert len(seeds) == 12
    assert all(a.info["pool"] == "validation" for a in seeds)
    assert {a.info["sample_kind"] for a in seeds} == {"strain_0.92", "strain_1.08"}


def test_tier2_hot_md_not_implemented(tmp_path: Path) -> None:
    p1 = make_dummy_seed(tmp_path, "x")
    with pytest.raises(NotImplementedError):
        build_tier2_indtest_seeds(
            seed_phases={"x": str(p1)},
            pressures_gpa=[0],
            n_hot_per_phase=5,
        )
