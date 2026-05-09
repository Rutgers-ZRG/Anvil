"""Unit tests for the global_entropy port — math correctness on a small NaCl-like cell.

These tests do NOT require a real foundation MLIP; they verify the
FingerprintDataset arithmetic and the round-trip save/load.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_fingerprint_dataset_round_trip(tmp_path) -> None:
    pytest.importorskip("libfp")
    from anvil.al.generation.global_entropy import FingerprintDataset

    fp_dim = 16
    ds = FingerprintDataset(fp_dim, reg=1e-3)
    rng = np.random.default_rng(42)
    fps = rng.normal(size=(20, fp_dim))
    ds.add(fps)

    Sigma, mu, n = ds.get_stats()
    assert n == 20
    assert Sigma.shape == (fp_dim, fp_dim)
    # Sigma is symmetric PSD (with reg)
    assert np.allclose(Sigma, Sigma.T)
    eig = np.linalg.eigvalsh(Sigma)
    assert eig.min() > 0  # reg ensures positive-definite

    path = tmp_path / "ds.npz"
    ds.save(str(path))
    rt = FingerprintDataset.load(str(path))
    assert rt.count == ds.count
    assert np.allclose(rt.sum_fp, ds.sum_fp)


def test_log_det_increases_with_diverse_fps() -> None:
    pytest.importorskip("libfp")
    from anvil.al.generation.global_entropy import FingerprintDataset

    fp_dim = 8
    ds = FingerprintDataset(fp_dim, reg=1e-3)

    # Add a tight cluster of FPs
    rng = np.random.default_rng(0)
    tight = rng.normal(scale=0.01, size=(10, fp_dim))
    ds.add(tight)
    base_logdet = ds.log_det()

    # A "diverse" extra fp far from the cluster should INCREASE log det
    diverse = np.full(fp_dim, 5.0)
    new_logdet = ds.log_det(extra_fps=diverse)
    assert new_logdet > base_logdet

    # A "redundant" extra fp similar to existing should change log det less
    redundant = rng.normal(scale=0.01, size=(fp_dim,))
    redundant_logdet = ds.log_det(extra_fps=redundant)
    assert (new_logdet - base_logdet) > (redundant_logdet - base_logdet)


def test_get_stats_with_does_not_mutate() -> None:
    pytest.importorskip("libfp")
    from anvil.al.generation.global_entropy import FingerprintDataset

    fp_dim = 4
    ds = FingerprintDataset(fp_dim)
    ds.add(np.ones((3, fp_dim)))
    original_count = ds.count
    original_sum = ds.sum_fp.copy()

    # Calling get_stats_with should not mutate the dataset
    ds.get_stats_with(np.ones((5, fp_dim)) * 99)
    assert ds.count == original_count
    assert np.allclose(ds.sum_fp, original_sum)
