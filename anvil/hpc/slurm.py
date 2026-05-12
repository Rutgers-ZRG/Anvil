"""Slurm submission, polling, and queue-cap-aware throttling.

See DESIGN.md §4.10, §10 (queue-cap truncation lesson).
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SlurmJobSpec:
    name: str
    partition: str
    n_tasks: int = 1
    n_nodes: int = 1
    cpus_per_task: int = 1
    mem: str = "8G"
    time: str = "01:00:00"
    output: str = "%x_%j.out"
    error: str = "%x_%j.err"
    gres: str = ""                       # e.g. "gpu:1"
    chdir: str = ""                       # explicit chdir (avoid amarel3 /cache/home)
    extra_directives: list[str] = field(default_factory=list)

    def directives(self) -> list[str]:
        d = [
            f"#SBATCH --job-name={self.name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --nodes={self.n_nodes}",
            f"#SBATCH --ntasks={self.n_tasks}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --mem={self.mem}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --output={self.output}",
            f"#SBATCH --error={self.error}",
        ]
        if self.gres:
            d.append(f"#SBATCH --gres={self.gres}")
        if self.chdir:
            d.append(f"#SBATCH --chdir={self.chdir}")
        d.extend(self.extra_directives)
        return d

    def render(self, body: str) -> str:
        """Render full sbatch script: shebang + directives + body.

        Uses `#!/bin/bash -l` to make it a login shell so /etc/profile.d/*.sh
        runs (provides `module`, `srun`, etc. on amarel3 where the default
        Slurm shell skips these). Also explicitly sources lmod for safety.
        """
        preamble = [
            "# Source modules for clusters where non-login shells skip it",
            "source /etc/profile.d/lmod.sh 2>/dev/null || true",
            "source /etc/profile.d/modules.sh 2>/dev/null || true",
        ]
        return "\n".join(["#!/bin/bash -l", *self.directives(),
                          "", *preamble, "", body, ""])


class SlurmError(RuntimeError):
    """Raised on sbatch / squeue / sacct failures."""


def _run(cmd: list[str], *, ssh_host: Optional[str] = None) -> str:
    """Run a command locally or over SSH, return stdout."""
    if ssh_host:
        cmd = ["ssh", "-o", "BatchMode=yes", ssh_host, " ".join(shlex.quote(c) for c in cmd)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SlurmError(
            f"Command failed: {' '.join(cmd)!s}\nstdout: {p.stdout}\nstderr: {p.stderr}"
        )
    return p.stdout


def queue_size(user: str | None = None, *, ssh_host: Optional[str] = None) -> int:
    """Return number of pending+running jobs for `user` (default $USER)."""
    user = user or os.environ.get("USER", "")
    if not user:
        return 0
    out = _run(["squeue", "-u", user, "-h", "-o", "%i"], ssh_host=ssh_host)
    return sum(1 for line in out.splitlines() if line.strip())


def submit(
    script_path: str | Path,
    *,
    ssh_host: Optional[str] = None,
    parsable: bool = True,
) -> str:
    """Submit one job; return Slurm job ID."""
    cmd = ["sbatch"]
    if parsable:
        cmd.append("--parsable")
    cmd.append(str(script_path))
    out = _run(cmd, ssh_host=ssh_host)
    if parsable:
        return out.strip().split(";")[0]
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise SlurmError(f"Could not parse job ID from sbatch output:\n{out}")
    return m.group(1)


def submit_throttled(
    script_paths: list[str | Path],
    *,
    user: str | None = None,
    max_concurrent: int = 400,
    backoff_seconds: float = 30.0,
    ssh_host: Optional[str] = None,
    max_iterations: int = 10_000,
) -> list[str]:
    """Submit many jobs while keeping queue size under max_concurrent.

    Codifies the lesson learned: amareln's 500-job cap silently truncates
    bulk sbatch when exceeded. We monitor queue_size() and back off.
    """
    job_ids: list[str] = []
    iters = 0
    for sp in script_paths:
        # Wait until queue has room
        while queue_size(user, ssh_host=ssh_host) >= max_concurrent:
            iters += 1
            if iters > max_iterations:
                raise SlurmError(
                    f"Queue stayed >= {max_concurrent} for {max_iterations} iterations"
                )
            time.sleep(backoff_seconds)
        jid = submit(sp, ssh_host=ssh_host)
        job_ids.append(jid)
    return job_ids


def poll(job_ids: list[str], *, ssh_host: Optional[str] = None) -> dict[str, str]:
    """Return {job_id: state} via squeue (live) + sacct (terminal)."""
    out: dict[str, str] = {}
    if not job_ids:
        return out

    # squeue first (live jobs)
    try:
        squeue_out = _run(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i %T"],
            ssh_host=ssh_host,
        )
        for line in squeue_out.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    except SlurmError:
        pass  # job may have completed → squeue returns error

    # sacct for missing IDs
    missing = [j for j in job_ids if j not in out]
    if missing:
        try:
            sacct_out = _run(
                ["sacct", "-X", "-j", ",".join(missing), "-n", "-o", "JobID,State"],
                ssh_host=ssh_host,
            )
            for line in sacct_out.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    # State may have prefixes like "CANCELLED by 12345"
                    out[parts[0]] = parts[1]
        except SlurmError:
            for j in missing:
                out[j] = "UNKNOWN"

    # Anyone still missing
    for j in job_ids:
        out.setdefault(j, "UNKNOWN")
    return out
