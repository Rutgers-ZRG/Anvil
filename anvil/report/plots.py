"""Plotting helpers.

Style follows npj Computational Materials guidelines (DESIGN.md §9 #14):
  - Colorblind-safe palette (Wong scheme by default).
  - Larger font sizes (10 pt minimum, 12 pt axis labels).
  - Fewer panels per figure; export as PDF + PNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ase import Atoms
from ase.calculators.calculator import Calculator


# Wong colorblind-safe palette
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "lblue": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


def plot_learning_curves(
    history: list[dict],
    out: str | Path,
    *,
    metrics: Iterable[str] = ("E_MAE", "F_MAE", "S_MAE"),
) -> Path:
    """Per-round validation MAE."""
    raise NotImplementedError("week 4")


def plot_parity(
    ref_atoms: list[Atoms],
    pred_calc: Calculator,
    out: str | Path,
) -> Path:
    """E / F / S parity plots, ref vs pred."""
    raise NotImplementedError("week 4")


def plot_phase_coverage(
    val_atoms: list[Atoms],
    pred_calc: Calculator,
    out: str | Path,
) -> Path:
    """Heatmap (phase × P × T) colored by E_MAE."""
    raise NotImplementedError("week 4")


def plot_eos(
    seed_phases: dict[str, str],
    pred_calc: Calculator,
    dft_eos: dict | None,
    out: str | Path,
) -> Path:
    """V(P) curves per phase: MLIP vs DFT seed."""
    raise NotImplementedError("week 4")


def plot_pool_composition(
    history: list[dict],
    out: str | Path,
) -> Path:
    """Per-round stacked bar chart of A/B/C pool contributions."""
    raise NotImplementedError("week 4")
