"""Smoke test — verifies the package imports cleanly.

Once week-1 implementation lands, this file gains real unit tests for:
  - config schema validation
  - state machine transitions
  - cluster detection
"""

from __future__ import annotations


def test_import_anvil() -> None:
    import anvil
    assert anvil.__version__ == "0.0.1"


def test_import_subpackages() -> None:
    import anvil.cli
    import anvil.config
    import anvil.state
    import anvil.orchestrator
    import anvil.dft
    import anvil.ml
    import anvil.al
    import anvil.al.generation
    import anvil.validation
    import anvil.hpc
    import anvil.report
