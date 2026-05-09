"""Dataset-aware global entropy calculator using libfp fingerprints.

PORTED from /Users/li/dev/RA/mlip-active-learn/global_entropy_calculator.py.
The only adaptation vs. the original is the libfp fallback: if libfp is not
installed as a top-level package, fall back to reformpy's pure-Python
implementation (reformpy.libfppy). All other code is verbatim.

The calculator wraps any ASE calculator and adds a force term that drives
the current configuration to maximize log det(Sigma) of the feature
covariance matrix of the entire dataset (existing + current config).

Two modes:
  - per_atom:   covariance over individual atomic fingerprints -> disordered structures
  - per_config: covariance over config-mean fingerprints -> ordered structures

References:
  - Karabin & Perez, J. Chem. Phys. 153, 094110 (2020) — local entropy
  - Subramanyam & Perez, npj Comput. Mater. 11:218 (2025) — global log det
  - This work: dataset-aware global log det folded into Anvil's three-pool AL
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

try:
    import libfp
except ImportError:  # pragma: no cover — fallback path
    try:
        from reformpy import libfppy as libfp
    except ImportError as exc:
        raise ImportError(
            "Anvil global_entropy requires either libfp (top-level package) "
            "or reformpy (which provides reformpy.libfppy as a fallback). "
            "Install one of them."
        ) from exc


class FingerprintDataset:
    """Tracks sufficient statistics for incremental covariance computation.

    Stores (sum, sum_outer, count) so the covariance can be computed without
    storing all individual fingerprints.  O(d^2) memory regardless of dataset size.

    Sigma = sum_outer/count - mu*mu^T + reg*I
    where mu = sum_fp/count.
    """

    def __init__(self, fp_dim: int, reg: float = 1e-3) -> None:
        self.fp_dim = fp_dim
        self.reg = reg
        self.sum_fp = np.zeros(fp_dim)
        self.sum_outer = np.zeros((fp_dim, fp_dim))
        self.count = 0

    def add(self, fps: np.ndarray) -> None:
        """Permanently add fingerprints to the dataset.

        Args:
            fps: shape (n, fp_dim) or (fp_dim,) for a single vector
        """
        fps = np.atleast_2d(np.asarray(fps, dtype=np.float64))
        self.sum_fp += fps.sum(axis=0)
        self.sum_outer += fps.T @ fps
        self.count += fps.shape[0]

    def get_stats(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Return (Sigma, mu, N) for the current dataset."""
        return self._compute_stats(self.sum_fp, self.sum_outer, self.count)

    def get_stats_with(self, extra_fps: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Return (Sigma, mu, N) temporarily including extra_fps.

        The dataset itself is NOT modified.
        """
        extra = np.atleast_2d(np.asarray(extra_fps, dtype=np.float64))
        s = self.sum_fp + extra.sum(axis=0)
        q = self.sum_outer + extra.T @ extra
        n = self.count + extra.shape[0]
        return self._compute_stats(s, q, n)

    def _compute_stats(self, s: np.ndarray, q: np.ndarray, n: int):
        if n < 2:
            return self.reg * np.eye(self.fp_dim), np.zeros(self.fp_dim), n
        mu = s / n
        Sigma = q / n - np.outer(mu, mu) + self.reg * np.eye(self.fp_dim)
        return Sigma, mu, n

    def log_det(self, extra_fps: np.ndarray | None = None) -> float:
        """Compute log det(Sigma), optionally including extra fingerprints."""
        if extra_fps is not None:
            Sigma, _, _ = self.get_stats_with(extra_fps)
        else:
            Sigma, _, _ = self.get_stats()
        sign, logdet = np.linalg.slogdet(Sigma)
        return float(logdet) if sign > 0 else -1e10

    def save(self, path: str) -> None:
        """Save dataset state to .npz file."""
        np.savez(path,
                 sum_fp=self.sum_fp,
                 sum_outer=self.sum_outer,
                 count=np.array([self.count]),
                 fp_dim=np.array([self.fp_dim]),
                 reg=np.array([self.reg]))

    @classmethod
    def load(cls, path: str) -> "FingerprintDataset":
        """Load dataset state from .npz file."""
        data = np.load(path)
        ds = cls(int(data['fp_dim'][0]), float(data['reg'][0]))
        ds.sum_fp = data['sum_fp']
        ds.sum_outer = data['sum_outer']
        ds.count = int(data['count'][0])
        return ds


def atoms_to_libfp_cell(atoms):
    """Convert ASE Atoms to libfp cell tuple (lat, rxyz, types, znucl)."""
    lat = np.array(atoms.cell[:], dtype=np.float64)
    rxyz = np.array(atoms.get_positions(), dtype=np.float64)
    atomic_numbers = atoms.get_atomic_numbers()
    unique_z = sorted(set(atomic_numbers))
    z_to_type = {z: i + 1 for i, z in enumerate(unique_z)}
    types = np.array([z_to_type[z] for z in atomic_numbers], dtype=np.int32)
    znucl = np.array(unique_z, dtype=np.int32)
    return (lat, rxyz, types, znucl)


def compute_fingerprints(atoms, cutoff, natx, need_grad=False, need_stress=False):
    """Compute libfp fingerprints and optionally derivatives.

    Returns:
        dict with keys 'fp', 'dfp' (or None), 'dfpe' (or None)
        fp:   (nat, natx)
        dfp:  (nat, nat, 3, natx)
        dfpe: (nat, 6, natx)
    """
    cell = atoms_to_libfp_cell(atoms)
    if need_grad:
        result = libfp.get_dfp(cell, cutoff=cutoff, natx=natx,
                               include_stress=need_stress, log=False)
        if need_stress:
            fp, dfp, dfpe = result
        else:
            fp, dfp = result
            dfpe = None
    else:
        fp = libfp.get_lfp(cell, cutoff=cutoff, natx=natx, log=False)
        dfp = None
        dfpe = None

    return {
        'fp': np.asarray(fp, dtype=np.float64),
        'dfp': np.asarray(dfp, dtype=np.float64) if dfp is not None else None,
        'dfpe': np.asarray(dfpe, dtype=np.float64) if dfpe is not None else None,
    }


class GlobalEntropyCalculator(Calculator):
    """ASE calculator: E = E_base - k * log det Sigma(D union current).

    The global covariance Sigma is computed over the existing dataset D
    plus the fingerprints of the current configuration. During MD, the
    entropy term drives atoms toward regions of fingerprint space not
    yet covered by D, preventing self-averaging across configurations.

    Gradient derivation (per-atom mode):
        d(log det Sigma)/dr_{i,k} = (2/N) sum_j w_j^T dfp[j,i,k,:]
        where w_j = Sigma^{-1} (fp_j - mu),  j in current config

    Gradient derivation (per-config mode):
        d(log det Sigma)/dr_{i,k} = (2/(N*A)) w_X^T sum_j dfp[j,i,k,:]
        where w_X = Sigma^{-1} (X - mu),  X = mean(fp), A = n_atoms
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, calculator, dataset, k_factor=5.0, cutoff=5.0,
                 natx=50, mode='per_atom', **kwargs):
        """
        Args:
            calculator: Base ASE calculator (e.g. Allegro / NequIP / MatterSim)
            dataset:    FingerprintDataset instance (shared across MD runs)
            k_factor:   Entropy scaling weight (0 = pure base calculator)
            cutoff:     libfp fingerprint cutoff (Angstroms)
            natx:       Max neighbors / fingerprint dimension
            mode:       'per_atom' or 'per_config'
        """
        super().__init__(**kwargs)
        self.base_calc = calculator
        self.dataset = dataset
        self.k_factor = k_factor
        self.cutoff = cutoff
        self.natx = natx
        self.mode = mode

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        # --- Base calculator ---
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.base_calc
        base_energy = atoms_copy.get_potential_energy()
        base_forces = atoms_copy.get_forces()
        base_stress = atoms_copy.get_stress()

        # Short-circuit if k=0
        if self.k_factor == 0:
            self.results['energy'] = base_energy
            self.results['forces'] = base_forces
            self.results['stress'] = base_stress
            return

        # --- Fingerprints ---
        need_grad = ('forces' in properties or 'stress' in properties)
        fp_data = compute_fingerprints(
            atoms, self.cutoff, self.natx,
            need_grad=need_grad, need_stress=('stress' in properties),
        )
        fp = fp_data['fp']          # (nat, d)
        dfp = fp_data['dfp']        # (nat, nat, 3, d) or None
        dfpe = fp_data['dfpe']      # (nat, 6, d) or None
        nat = fp.shape[0]
        d = fp.shape[1]

        # --- Covariance including current config ---
        if self.mode == 'per_atom':
            extra = fp                                    # (nat, d)
        else:
            extra = fp.mean(axis=0, keepdims=True)        # (1, d)

        Sigma, mu, N = self.dataset.get_stats_with(extra)

        # --- log det Sigma ---
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            logdet = -1e10

        # --- Energy ---
        self.results['energy'] = base_energy - self.k_factor * logdet

        # --- Forces ---
        if dfp is not None:
            entropy_forces = self._entropy_forces(fp, dfp, Sigma, mu, N, nat)
            self.results['forces'] = base_forces + self.k_factor * entropy_forces
        else:
            self.results['forces'] = base_forces

        # --- Stress ---
        if dfpe is not None:
            entropy_stress = self._entropy_stress(fp, dfpe, Sigma, mu, N, nat)
            volume = atoms.get_volume()
            self.results['stress'] = base_stress - (self.k_factor / volume) * entropy_stress
        else:
            self.results['stress'] = base_stress

    # ------------------------------------------------------------------
    # Gradient kernels
    # ------------------------------------------------------------------

    def _entropy_forces(self, fp, dfp, Sigma, mu, N, nat):
        """Compute d(log det Sigma)/dr for forces.

        Returns shape (nat, 3) — entropy gradient to ADD to base forces.
        """
        if self.mode == 'per_atom':
            # w_j = Sigma^{-1} (fp_j - mu),  shape (nat, d)
            centered = fp - mu                                      # (nat, d)
            w = np.linalg.solve(Sigma, centered.T).T                # (nat, d)
            # forces[i,k] = (2/N) sum_j w[j,m] * dfp[j,i,k,m]
            return (2.0 / N) * np.einsum('jm,jikm->ik', w, dfp)
        else:
            # Per-config mode
            X = fp.mean(axis=0)                                     # (d,)
            w_X = np.linalg.solve(Sigma, X - mu)                    # (d,)
            dfp_sum_j = dfp.sum(axis=0)                             # (nat, 3, d)
            return (2.0 / (N * nat)) * np.einsum('m,ikm->ik', w_X, dfp_sum_j)

    def _entropy_stress(self, fp, dfpe, Sigma, mu, N, nat):
        """Compute d(log det Sigma)/d_epsilon for stress.

        Returns shape (6,) in Voigt notation.
        """
        if self.mode == 'per_atom':
            centered = fp - mu
            w = np.linalg.solve(Sigma, centered.T).T                # (nat, d)
            # stress[c] = (2/N) sum_j w[j,m] * dfpe[j,c,m]
            return (2.0 / N) * np.einsum('jm,jcm->c', w, dfpe)
        else:
            X = fp.mean(axis=0)
            w_X = np.linalg.solve(Sigma, X - mu)
            dfpe_sum = dfpe.sum(axis=0)                             # (6, d)
            return (2.0 / (N * nat)) * np.einsum('m,cm->c', w_X, dfpe_sum)

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def add_config(self, atoms):
        """Permanently add a configuration's fingerprints to the dataset."""
        fp_data = compute_fingerprints(atoms, self.cutoff, self.natx)
        fp = fp_data['fp']
        if self.mode == 'per_atom':
            self.dataset.add(fp)
        else:
            self.dataset.add(fp.mean(axis=0, keepdims=True))

    def entropy_gain(self, atoms):
        """Compute Delta(log det Sigma) if this config were added.

        Returns:
            float: log det(Sigma_new) - log det(Sigma_current)
        """
        fp_data = compute_fingerprints(atoms, self.cutoff, self.natx)
        fp = fp_data['fp']
        current = self.dataset.log_det()
        if self.mode == 'per_atom':
            proposed = self.dataset.log_det(fp)
        else:
            proposed = self.dataset.log_det(fp.mean(axis=0, keepdims=True))
        return proposed - current
