"""Command-line interface for Anvil.

See DESIGN.md §4.1 for the full UX spec.

Usage:
    anvil submit  --config <yaml>     # cold start
    anvil resume  <run_id|run_dir>    # warm start from checkpoint
    anvil status  [<run_id|run_dir>]  # state machine inspection
    anvil report  [<run_id|run_dir>]  # generate HTML report (week 4)
    anvil cancel  <run_id|run_dir>    # graceful shutdown (week 2)
    anvil daemon  <run_id|run_dir>    # long-running orchestrator (week 2)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import click


def _resolve_run_dir(run_ref: str | None) -> Path:
    """Resolve a run_id, run_dir, or system name to a run_dir.

    Resolution order:
      1. If run_ref is an existing path, use it.
      2. If `.anvil/<run_ref>` symlink exists in cwd, follow it.
      3. If run_ref looks like a run_id (contains "-"), search ${SCRATCH}/
         anvil_runs/*/<run_ref>.
      4. Otherwise, treat as system name and follow `.anvil/<system>` symlink.
    """
    if run_ref is None:
        # Pick the only entry in .anvil/ if exactly one
        anvil_dir = Path.cwd() / ".anvil"
        if not anvil_dir.exists():
            raise click.ClickException("No `.anvil/` symlink dir in cwd. Pass run_id.")
        entries = [p for p in anvil_dir.iterdir() if p.is_symlink()]
        if len(entries) != 1:
            raise click.ClickException(
                f"Found {len(entries)} runs under .anvil/; pass system or run_id explicitly."
            )
        return Path(os.readlink(entries[0]))

    p = Path(run_ref)
    if p.exists() and p.is_dir():
        return p

    sym = Path.cwd() / ".anvil" / run_ref
    if sym.is_symlink():
        return Path(os.readlink(sym))

    # Fallback: search scratch
    scratch_root = Path(os.environ.get("SCRATCH", "/scratch/lz432"))
    candidates = list((scratch_root / "anvil_runs").glob(f"*/{run_ref}"))
    if len(candidates) == 1:
        return candidates[0]

    raise click.ClickException(f"Could not resolve run reference {run_ref!r}.")


@click.group()
@click.version_option()
def main() -> None:
    """Anvil: closed-loop active learning for MLIPs."""


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to user yaml.")
@click.option("--cluster", default="amareln", show_default=True,
              help="Target HPC cluster (amareln | amarel3).")
@click.option("--ssh-host", default=None,
              help="If set, run sbatch over SSH (e.g. 'an' for amareln login alias).")
@click.option("--no-submit", is_flag=True,
              help="Generate run dir + VASP inputs but skip sbatch (smoke test).")
@click.option("--scratch-root", default=None,
              help="Override scratch root (defaults to cluster scratch_root).")
def submit(config: str, cluster: str, ssh_host: Optional[str],
           no_submit: bool, scratch_root: Optional[str]) -> None:
    """Cold-start a new Anvil run from the user yaml.

    Walks INIT → BOOTSTRAP_SEEDS → BOOTSTRAP_DFT_QUEUED in this single
    command (suitable for an `srun --time=72h` daemon job in production;
    can also be invoked from a login node for smoke testing).
    """
    from anvil.orchestrator import Orchestrator

    orch = Orchestrator.from_config_path(
        config, cluster=cluster, scratch_root=scratch_root,
    )
    click.echo(f"Allocated run_id: {orch.ckpt.run_id}")
    click.echo(f"Run dir:          {orch.run_dir}")
    click.echo(f"Cluster:          {cluster}")
    click.echo(f"State:            {orch.ckpt.state.value}")

    orch.bootstrap(ssh_host=ssh_host, do_submit=not no_submit)
    click.echo(f"Final state:      {orch.ckpt.state.value}")
    click.echo(f"DFT calls queued: {orch.ckpt.dft_calls_used}")
    if orch.ckpt.pending_job_ids:
        click.echo(f"Slurm job IDs:    "
                   f"{orch.ckpt.pending_job_ids[0]}..{orch.ckpt.pending_job_ids[-1]} "
                   f"({len(orch.ckpt.pending_job_ids)} jobs)")
    click.echo(f"\nResume with: anvil resume {orch.run_dir}")


@main.command()
@click.argument("run_ref", required=False)
def resume(run_ref: Optional[str]) -> None:
    """Warm-start an existing run from its last checkpoint."""
    from anvil.orchestrator import Orchestrator
    run_dir = _resolve_run_dir(run_ref)
    orch = Orchestrator.resume(run_dir)
    click.echo(f"Resumed at state: {orch.ckpt.state.value}")
    click.echo(f"Run dir:          {orch.run_dir}")
    click.echo(f"DFT calls used:   {orch.ckpt.dft_calls_used}")


@main.command()
@click.argument("run_ref", required=False)
def status(run_ref: Optional[str]) -> None:
    """Print the current state, round number, DFT calls used, and pool sizes."""
    from anvil.state import load_checkpoint
    run_dir = _resolve_run_dir(run_ref)
    ckpt = load_checkpoint(run_dir)
    click.echo(json.dumps({
        "run_id": ckpt.run_id,
        "system": ckpt.system,
        "state": ckpt.state.value,
        "round": ckpt.round,
        "dft_calls_used": ckpt.dft_calls_used,
        "pending_jobs": len(ckpt.pending_job_ids),
        "cluster": ckpt.cluster,
        "run_dir": str(run_dir),
        "last_updated": ckpt.last_updated_iso,
    }, indent=2))


@main.command()
@click.argument("run_ref", required=False)
def report(run_ref: Optional[str]) -> None:
    """Generate the final HTML report (week 4)."""
    raise click.ClickException("report: not implemented in v1 piece-3 (week 4)")


@main.command()
@click.argument("run_ref")
def cancel(run_ref: str) -> None:
    """Graceful shutdown: cancel pending Slurm jobs (week 2)."""
    raise click.ClickException("cancel: not implemented in v1 piece-3 (week 2)")


@main.command()
@click.argument("run_ref")
def daemon(run_ref: str) -> None:
    """Long-running orchestrator process (week 2)."""
    raise click.ClickException("daemon: not implemented in v1 piece-3 (week 2)")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
