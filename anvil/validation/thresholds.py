"""Default Tier-2 pass/fail thresholds. See DESIGN.md §5.2."""

from __future__ import annotations

# meV/atom, meV/Å, meV/Å³
DEFAULT_THRESHOLDS: dict[str, float] = {
    "E_MAE": 10.0,
    "F_MAE": 100.0,
    "S_MAE": 10.0,
}


def check_thresholds(metrics: dict, thresholds: dict | None = None) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failed_keys)."""
    raise NotImplementedError("week 1")
