"""TorchScript compilation of trained nequip models for production deployment.

See DESIGN.md §4.5.
"""

from __future__ import annotations

from pathlib import Path


def compile_to_torchscript(model_dir: str | Path, target: str = "ase") -> Path:
    """Run `nequip-compile --mode torchscript --device cpu --target ase`.

    Returns path to the compiled .nequip.pth.

    Note: nequip-deploy was removed in 0.16.x; we use nequip-compile.
    """
    raise NotImplementedError("week 1")
