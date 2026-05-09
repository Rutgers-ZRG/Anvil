"""Command-line interface for Anvil.

See DESIGN.md §4.1 for the full UX spec.

Usage:
    anvil submit  --config <yaml>     # cold start
    anvil resume  <run_id>            # warm start from checkpoint
    anvil status  [<run_id>]          # state machine inspection
    anvil report  [<run_id>]          # generate HTML report
    anvil cancel  <run_id>            # graceful shutdown (cancels SLURM jobs)
    anvil daemon  <run_id>            # long-running orchestrator (srun job)
"""

from __future__ import annotations

import sys
from typing import Optional

import click


@click.group()
@click.version_option()
def main() -> None:
    """Anvil: closed-loop active learning for MLIPs."""


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to user yaml.")
def submit(config: str) -> None:
    """Cold-start a new Anvil run from the user yaml.

    Validates config, allocates a run_id, sets up the run directory under
    $SCRATCH/anvil_runs/<system>/<run_id>/, creates a .anvil/<system> symlink
    in cwd for `anvil status` / `anvil report` resolution, and submits the
    daemon job.
    """
    raise NotImplementedError("week 1: see DESIGN.md §4.1, §3.5 round 0")


@main.command()
@click.argument("run_id")
def resume(run_id: str) -> None:
    """Warm-start an existing run from its last checkpoint.

    Reads .anvil/state.json, jumps to the matching state, idempotently
    re-runs incomplete steps. Any state can be re-entered without corruption.
    """
    raise NotImplementedError("week 1: see DESIGN.md §4.2 state machine")


@main.command()
@click.argument("run_id", required=False)
def status(run_id: Optional[str]) -> None:
    """Print the current state, round number, DFT calls used, and pool sizes.

    If run_id omitted, resolves via .anvil symlink in cwd.
    """
    raise NotImplementedError("week 1: see DESIGN.md §4.2")


@main.command()
@click.argument("run_id", required=False)
def report(run_id: Optional[str]) -> None:
    """Generate the final HTML report for the run.

    Validation MAEs, parity plots, EOS, phase-coverage heatmap, σ trajectory.
    """
    raise NotImplementedError("week 4: see DESIGN.md §4.12, §5")


@main.command()
@click.argument("run_id")
def cancel(run_id: str) -> None:
    """Graceful shutdown: cancel pending Slurm jobs, mark run as CANCELLED."""
    raise NotImplementedError("week 1: see DESIGN.md §4.10")


@main.command()
@click.argument("run_id")
def daemon(run_id: str) -> None:
    """Long-running orchestrator process. Normally invoked from inside a Slurm
    srun --time=72h job, NOT directly by the user. Drives the state machine
    until DONE, PAUSED_PHYSICS, or CANCELLED.
    """
    raise NotImplementedError("week 1: see DESIGN.md §4.2")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
