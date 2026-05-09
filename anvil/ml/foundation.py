"""Foundation MLIP registry. v1 supports Allegro-OAM-L only.

See DESIGN.md §4.4, §9 decision #4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ase.calculators.calculator import Calculator


@dataclass
class FoundationConfig:
    name: str
    checkpoint_paths: dict[str, str]      # cluster_name -> path
    supported_elements: set[int] = field(default_factory=set)


class FoundationRegistry:
    """Registry of available foundation MLIPs.

    Pre-flight check: framework refuses to start if user's elements aren't
    all in foundation.supported_elements. Suggests `from_dft_only: true`
    workaround in v2.
    """

    @classmethod
    def get(cls, name: str) -> FoundationConfig:
        raise NotImplementedError("week 1")

    @classmethod
    def make_calc(
        cls,
        name: str,
        cluster: str = "amareln",
        device: str = "cuda",
    ) -> Calculator:
        """Load .nequip.pth and return ASE NequIPCalculator on the given device."""
        raise NotImplementedError("week 1")


# Registry — populated at import time.
ALLEGRO_OAM_L = FoundationConfig(
    name="allegro-oam-l-foundation",
    checkpoint_paths={
        "amareln": "/scratch/lz432/allegro_finetune/allegro-oam-l-foundation.nequip.pth",
        "amarel3": "/scratch/lz432/allegro_finetune/allegro-oam-l-foundation.nequip.pth",
    },
    supported_elements=set(),  # populated from model metadata at first use
)


def bootstrap_foundation(target_cluster: str) -> str:
    """One-time scp of the foundation model from amareln to target_cluster
    if missing on target. Idempotent (skips if content hash matches).

    See DESIGN.md §9 decision #9.
    """
    raise NotImplementedError("week 1")
