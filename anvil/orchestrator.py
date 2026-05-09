"""Main state-machine orchestrator. Drives the closed-loop AL.

See DESIGN.md §3.5 (algorithm) and §4.2 (state machine).
"""

from __future__ import annotations

from pathlib import Path

from anvil.config import AnvilConfig
from anvil.state import AnvilState, StateCheckpoint


class Orchestrator:
    """Main loop. Runs as `anvil daemon` inside a Slurm srun job.

    Reads StateCheckpoint, advances one transition, writes checkpoint, repeats
    until terminal state.
    """

    def __init__(self, config: AnvilConfig, run_dir: str | Path) -> None:
        self.config = config
        self.run_dir = Path(run_dir)

    def run(self) -> AnvilState:
        """Run until terminal state. Returns the terminal state.

        Steps:
          0. Initialize: ensure run_dir, .anvil/, validate config.
          1. Bootstrap (rounds 0): seeds, foundation MD, anchor + entropy + reform pools.
          2. Loop: candidate generation, DFT label, retrain, validate.
          3. Stop when budget exhausted or σ plateau or Tier-1 fail.
          4. Final compile + validate + report.
        """
        raise NotImplementedError("week 1: see DESIGN.md §3.5, §4.2")

    def step(self) -> AnvilState:
        """Advance one state transition. Returns the new state."""
        raise NotImplementedError("week 1: see DESIGN.md §4.2")
