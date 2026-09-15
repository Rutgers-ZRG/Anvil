"""Command-line interface for Anvil.

See DESIGN.md §4.1 for the full UX spec.

Usage:
    anvil submit  --config <yaml>     # cold start
    anvil resume  <run_id|run_dir>    # warm start from checkpoint
    anvil status  [<run_id|run_dir>]  # state machine inspection
    anvil collect [<run_id|run_dir>]  # collect finished DFT -> train/val.xyz
    anvil daemon  [<run_id|run_dir>]  # drive the AL loop to completion
    anvil cancel  <run_id|run_dir>    # cancel the run's pending Slurm jobs
    anvil report  [<run_id|run_dir>]  # generate HTML report (not implemented)
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
@click.option("--skip-pools", default="",
              help="Comma-separated pools to skip, e.g. 'A,B'. Pools A and B "
                   "run foundation-MLIP MD in this process and need a GPU — "
                   "skip them to generate only anchors + validation.")
def submit(config: str, cluster: str, ssh_host: Optional[str],
           no_submit: bool, scratch_root: Optional[str], skip_pools: str) -> None:
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

    pools = tuple(p.strip().upper() for p in skip_pools.split(",") if p.strip())
    if pools:
        click.echo(f"Skipping pools:   {', '.join(pools)}")
    orch.bootstrap(ssh_host=ssh_host, do_submit=not no_submit, skip_pools=pools)
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
@click.option("--wait/--no-wait", default=True, show_default=True,
              help="Wait for the checkpoint's pending Slurm jobs to finish first.")
@click.option("--poll-seconds", default=60.0, show_default=True)
@click.option("--fs-settle", default=0.0, show_default=True,
              help="Pause this long after the jobs finish before reading their "
                   "output. Needed where compute nodes see a cached view of "
                   "the shared filesystem (amarel3).")
def collect(run_ref: Optional[str], wait: bool, poll_seconds: float,
            fs_settle: float) -> None:
    """Collect finished DFT jobs into train.xyz / val.xyz.

    Works at either labeling state: BOOTSTRAP_DFT_QUEUED (bootstrap pools +
    validation) or ROUND_DFT_QUEUED (the current round's batch).
    """
    from anvil.driver import DriverOptions, wait_for_jobs, _settle
    from anvil.orchestrator import Orchestrator
    from anvil.state import AnvilState

    run_dir = _resolve_run_dir(run_ref)
    orch = Orchestrator.resume(run_dir)
    state = orch.ckpt.state
    if state not in (AnvilState.BOOTSTRAP_DFT_QUEUED, AnvilState.ROUND_DFT_QUEUED):
        raise click.ClickException(
            f"Nothing to collect at state {state.value}. `anvil collect` applies "
            f"at bootstrap_dft_queued or round_dft_queued."
        )

    opts = DriverOptions(poll_seconds=poll_seconds, fs_settle_seconds=fs_settle,
                         log=click.echo)
    if wait:
        wait_for_jobs(orch, opts)
    _settle(opts)

    if state is AnvilState.BOOTSTRAP_DFT_QUEUED:
        orch.bootstrap_dft_done()
        orch.val_dft_done()
    else:
        orch.round_dft_done()

    click.echo(f"State:   {orch.ckpt.state.value}")
    for name in ("train.xyz", "val.xyz"):
        p = orch.run_dir / name
        if p.exists():
            from ase.io import read as ase_read
            click.echo(f"{name}: {len(ase_read(str(p), index=':', format='extxyz'))} frames")


@main.command()
@click.argument("run_ref", required=False)
def report(run_ref: Optional[str]) -> None:
    """Generate the final HTML report (week 4)."""
    raise click.ClickException("report: not implemented in v1 piece-3 (week 4)")


@main.command()
@click.argument("run_ref")
@click.option("--ssh-host", default=None, help="Run scancel over SSH.")
def cancel(run_ref: str, ssh_host: Optional[str]) -> None:
    """Cancel the run's pending Slurm jobs."""
    from anvil.hpc.slurm import cancel as slurm_cancel
    from anvil.orchestrator import Orchestrator

    run_dir = _resolve_run_dir(run_ref)
    orch = Orchestrator.resume(run_dir)
    ids = slurm_cancel(orch.ckpt.pending_job_ids or [], ssh_host=ssh_host)
    if not ids:
        click.echo("No pending jobs recorded in the checkpoint.")
        return
    click.echo(f"Cancelled {len(ids)} job(s): {ids[0]}..{ids[-1]}")
    click.echo(f"State left at {orch.ckpt.state.value}; "
               f"`anvil daemon {run_dir}` resumes.")


@main.command()
@click.argument("run_ref", required=False)
@click.option("--poll-seconds", default=60.0, show_default=True,
              help="How often to check Slurm while waiting.")
@click.option("--max-rounds", default=None, type=int,
              help="Stop after this AL round (default: run until a stop "
                   "criterion fires).")
@click.option("--device", default="cuda", show_default=True,
              help="Device for the foundation / ensemble MLIP.")
@click.option("--skip-pools", default="",
              help="Comma-separated pools to skip, e.g. 'A,B'.")
@click.option("--ssh-host", default=None, help="Submit over SSH.")
@click.option("--fs-settle", default=0.0, show_default=True,
              help="Pause before reading job output (cached filesystems).")
@click.option("--train-time", default="12:00:00", show_default=True,
              help="Slurm walltime for each training job.")
def daemon(run_ref: Optional[str], poll_seconds: float, max_rounds: Optional[int],
           device: str, skip_pools: str, ssh_host: Optional[str],
           fs_settle: float, train_time: str) -> None:
    """Drive the AL loop: collect, train, validate, next round, until it stops.

    Holds a GPU whenever pools A/B or training run, so submit it as a job:

        sbatch -p gpu --gres=gpu:1 -t 24:00:00 --wrap "anvil daemon <run_dir>"

    Resumable: every step is a checkpointed transition, so re-running it
    continues from wherever the run currently is.
    """
    from anvil.driver import DriverOptions, run_loop
    from anvil.orchestrator import Orchestrator

    run_dir = _resolve_run_dir(run_ref)
    orch = Orchestrator.resume(run_dir)
    opts = DriverOptions(
        poll_seconds=poll_seconds,
        max_rounds=max_rounds,
        device=device,
        skip_pools=tuple(p.strip().upper() for p in skip_pools.split(",") if p.strip()),
        ssh_host=ssh_host,
        fs_settle_seconds=fs_settle,
        train_time=train_time,
        log=click.echo,
    )
    final = run_loop(orch, opts)
    click.echo(f"Final state: {final.value}")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
