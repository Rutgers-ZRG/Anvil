"""Three structure-generation pools. See DESIGN.md §3.1-3.4, §4.6.

Pool A — Per-regime entropy MD (local + global)         (workhorse for energy)
Pool B — Reform-relax (CRISP-style, FP-targeted)        (closes carbon F-gap)
Pool C — Anchor (foundation MLIP only, regime-aligned)  (task-aligned support)
"""
