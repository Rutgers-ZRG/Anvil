"""Default Tier-2 pass/fail thresholds. See DESIGN.md §5.2."""

from __future__ import annotations

# meV/atom, meV/Å, meV/Å³
DEFAULT_THRESHOLDS: dict[str, float] = {
    "E_MAE": 10.0,
    "F_MAE": 100.0,
    "S_MAE": 10.0,
}


def check_thresholds(
    metrics: dict, thresholds: dict | None = None,
) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failed_keys).

    Each threshold key (E_MAE / F_MAE / S_MAE) must be present in metrics
    with a value below its threshold. NaN counts as a failure.
    """
    import math
    thresholds = thresholds or DEFAULT_THRESHOLDS
    failed = []
    for k, ceiling in thresholds.items():
        v = metrics.get(k, float("nan"))
        if isinstance(v, float) and math.isnan(v):
            failed.append(k)
        elif v > ceiling:
            failed.append(k)
    return (not failed, failed)
