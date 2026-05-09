"""HTML report generator. See DESIGN.md §4.12."""

from __future__ import annotations

from pathlib import Path


def render_report(run_dir: str | Path, output_html: str | Path) -> Path:
    """Render the full HTML report.

    Sections:
      1. Header — system, run_id, terminal state, supported regime box.
      2. Validation — Tier-1 pass/fail, Tier-2 MAE table, parity plots.
      3. Learning curves — val MAE per round, σ trajectory if v2.
      4. Phase coverage — heatmap (phase × P × T) colored by E_MAE.
      5. EOS V(P) per phase — MLIP vs DFT seed.
      6. Pool composition — per-round A/B/C contributions.
      7. Provenance — config hash, pool hashes, ensemble hashes,
         npj-style methods paragraph autogen for paper draft.
    """
    raise NotImplementedError("week 4: see DESIGN.md §4.12")
