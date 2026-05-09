"""Environment hygiene for Slurm scripts.

Codifies the operational lessons from DESIGN.md §10:
  - `conda activate` clobbers $SIZE (and $STRIP). Use shadow-variable pattern:
    capture original env to __MS_* names BEFORE activate, then restore.
  - `nequip-deploy` was removed in nequip 0.16.x — use `nequip-compile`.
  - amarel3 default chdir is /cache/home (node-local, cleared post-job) —
    always pass --chdir=/scratch/... explicitly.
"""

from __future__ import annotations

# Variables we know conda activation overwrites with binary tool names
# (e.g., $SIZE becomes "x86_64-conda-linux-gnu-size"). Capture-and-restore
# these around the activate call.
SHADOW_VARS: tuple[str, ...] = ("SIZE", "STRIP", "AS", "AR", "LD", "NM", "RANLIB")


def shadow_env_activation_script(
    conda_env: str,
    user_vars_to_save: list[str] | None = None,
) -> str:
    """Return a bash snippet that activates conda_env without clobbering
    SHADOW_VARS or any user-specified variables.

    Pattern:
      __MS_SIZE="$SIZE"; __MS_STRIP="$STRIP"; ...
      eval "$(/path/to/conda shell.bash hook)"
      conda activate <env>
      SIZE="$__MS_SIZE"; STRIP="$__MS_STRIP"; ...
    """
    raise NotImplementedError("week 1: see DESIGN.md §10 SIZE clobber lesson")


def amarel3_chdir_directive(scratch_path: str) -> str:
    """Return the SBATCH --chdir directive needed on amarel3 to avoid the
    /cache/home node-local default that gets cleared post-job."""
    return f"#SBATCH --chdir={scratch_path}"
