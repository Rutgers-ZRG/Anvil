"""The loop driver — advances an Anvil run through the state machine.

`Orchestrator` exposes one method per state transition; something has to call
them in order, wait for Slurm in between, and stop when the stop criteria say
so. That is this module, and it is what `anvil daemon` runs.

Intended to run inside a long-lived allocation (it holds a GPU whenever pools
A/B or training are involved), not on a login node::

    sbatch --partition=gpu --gres=gpu:1 --time=24:00:00 \
        --wrap="anvil daemon <run_dir>"

Every step is a checkpointed transition, so killing the driver and starting it
again resumes where it left off. See DESIGN.md §3.5, §4.2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from anvil.orchestrator import Orchestrator
from anvil.state import TERMINAL_STATES, AnvilState

# States where the run is waiting on Slurm rather than on us.
_WAIT_STATES = {
    AnvilState.BOOTSTRAP_DFT_QUEUED,
    AnvilState.ROUND_DFT_QUEUED,
    AnvilState.BOOTSTRAP_TRAIN,
    AnvilState.ROUND_TRAIN,
}

_LIVE_JOB_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING",
                    "SUSPENDED", "RESIZING", "REQUEUED"}


@dataclass
class DriverOptions:
    """Knobs for one `run_loop` invocation."""

    poll_seconds: float = 60.0
    max_rounds: Optional[int] = None        # None = until a stop criterion fires
    device: str = "cuda"                    # foundation/ensemble MLIP device
    skip_pools: tuple[str, ...] = ()
    ssh_host: Optional[str] = None
    train_time: str = "12:00:00"
    fs_settle_seconds: float = 0.0          # see note in `_settle`
    max_wait_hours: float = 72.0
    log: Callable[[str], None] = print
    _t0: float = field(default_factory=time.time, repr=False)


def pending_job_ids(orch: Orchestrator) -> list[str]:
    """Real Slurm ids from the checkpoint ("DONE"/"" are skip markers)."""
    return [j for j in (orch.ckpt.pending_job_ids or [])
            if j and j != "DONE"]


def wait_for_jobs(orch: Orchestrator, opts: DriverOptions) -> None:
    """Block until none of the checkpoint's job ids are live."""
    from anvil.hpc.slurm import poll

    ids = pending_job_ids(orch)
    if not ids:
        return
    deadline = time.time() + opts.max_wait_hours * 3600
    while True:
        states = poll(ids, ssh_host=opts.ssh_host)
        live = [j for j, s in states.items() if s.upper() in _LIVE_JOB_STATES]
        if not live:
            opts.log(f"[driver] {len(ids)} job(s) finished")
            return
        if time.time() > deadline:
            raise TimeoutError(
                f"{len(live)} job(s) still live after {opts.max_wait_hours} h: "
                f"{live[:5]}"
            )
        opts.log(f"[driver] waiting on {len(live)}/{len(ids)} job(s)...")
        time.sleep(opts.poll_seconds)


def _settle(opts: DriverOptions) -> None:
    """Pause before reading job output.

    On clusters whose compute nodes see a cached view of the shared filesystem
    (amarel3 mounts /scratch through /scache), a job's OUTCAR can be absent
    from the submit host for a while after the job finishes. Collecting too
    early marks converged jobs as failed. Default 0 — set it per site.
    """
    if opts.fs_settle_seconds > 0:
        opts.log(f"[driver] settling {opts.fs_settle_seconds:.0f}s for the filesystem")
        time.sleep(opts.fs_settle_seconds)


def step(orch: Orchestrator, opts: DriverOptions) -> bool:
    """Advance the run by one transition. Returns False when it should stop."""
    state = orch.ckpt.state

    if state in TERMINAL_STATES:
        opts.log(f"[driver] terminal state {state.value}; nothing to do")
        return False

    if state in _WAIT_STATES:
        wait_for_jobs(orch, opts)
        _settle(opts)

    if state in (AnvilState.INIT, AnvilState.BOOTSTRAP_SEEDS):
        orch.bootstrap(ssh_host=opts.ssh_host, skip_pools=opts.skip_pools)

    elif state is AnvilState.BOOTSTRAP_DFT_QUEUED:
        orch.bootstrap_dft_done()
        orch.val_dft_done()

    elif state is AnvilState.VAL_DFT_DONE:
        orch.bootstrap_train(ssh_host=opts.ssh_host, time=opts.train_time)

    elif state is AnvilState.BOOTSTRAP_TRAIN:
        orch.bootstrap_trained()

    elif state is AnvilState.BOOTSTRAP_TRAINED:
        orch.round_validated(device=opts.device)

    elif state is AnvilState.ROUND_CANDIDATES:
        if opts.max_rounds is not None and orch.ckpt.round >= opts.max_rounds:
            opts.log(f"[driver] reached max_rounds={opts.max_rounds}; stopping")
            return False
        orch.round_candidates(skip_pools=opts.skip_pools, device=opts.device)

    elif state is AnvilState.ROUND_ACQUIRED:
        orch.round_dft_queued(ssh_host=opts.ssh_host)

    elif state is AnvilState.ROUND_DFT_QUEUED:
        orch.round_dft_done()

    elif state is AnvilState.ROUND_DFT_DONE:
        orch.round_train(ssh_host=opts.ssh_host, time=opts.train_time)

    elif state is AnvilState.ROUND_TRAIN:
        orch.round_trained()

    elif state is AnvilState.ROUND_TRAINED:
        orch.round_validated_and_decide(device=opts.device)

    elif state is AnvilState.FINAL_COMPILE:
        # The AL loop is finished; FINAL_COMPILE → FINAL_VALIDATE → DONE has
        # no implementation yet (see DESIGN.md milestones).
        opts.log(
            "[driver] stop criterion reached — run is at FINAL_COMPILE. "
            "Final compile/validate/report are not implemented; the trained "
            f"ensemble is under {orch.run_dir}."
        )
        return False

    else:
        opts.log(f"[driver] no handler for state {state.value}; stopping")
        return False

    opts.log(f"[driver] {state.value} → {orch.ckpt.state.value} "
             f"(round {orch.ckpt.round}, dft_calls {orch.ckpt.dft_calls_used})")
    return True


def run_loop(orch: Orchestrator, opts: Optional[DriverOptions] = None) -> AnvilState:
    """Drive the run until it stops. Returns the state it stopped in."""
    opts = opts or DriverOptions()
    opts.log(f"[driver] starting at {orch.ckpt.state.value}, run_dir={orch.run_dir}")
    while step(orch, opts):
        pass
    opts.log(f"[driver] stopped at {orch.ckpt.state.value}")
    return orch.ckpt.state
