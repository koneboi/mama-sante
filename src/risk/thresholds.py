"""Group-aware risk thresholds (fairness by design).

Thresholds are calibrated per group (region, wealth quartile) so that groups the
pooled threshold systematically under-flags get an earlier, lower decision
threshold, and over-flagged groups get a higher one. The table is produced by
``scripts/fairness.py`` and written to ``data/processed/group_thresholds.json``.

Combination rule for a patient belonging to two audited groups: take the
**minimum** of the available group thresholds (most sensitive group wins) so an
under-flagged group can never be delayed by the other group's threshold.
Missing entries / missing file / ``strategy="pooled"`` all fall back to the base
threshold, so the API degrades gracefully to the original behaviour.
"""
from __future__ import annotations

import json
import os

DEFAULT_BASE = 0.5
DEFAULT_BOUNDS = (0.02, 0.95)
STRATEGIES = ("pooled", "group")


def load(path: str) -> dict:
    """Load the threshold table; missing/corrupt file -> {} (fallback to pooled)."""
    try:
        with open(path) as f:
            table = json.load(f)
        return table if isinstance(table, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_threshold(
    region=None,
    wealth_quartile=None,
    rural_flag=None,
    table: dict | None = None,
    strategy: str = "pooled",
    base: float = DEFAULT_BASE,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
) -> float:
    """Return the risk threshold to apply for this patient."""
    if strategy != "group" or not table:
        return float(base)

    candidates: list[float] = []
    for col, value in (
        ("region", region),
        ("wealth_quartile", wealth_quartile),
        ("rural_flag", rural_flag),
    ):
        group_t = table.get(col) or {}
        if value is not None and str(value) in group_t:
            candidates.append(float(group_t[str(value)]))

    if not candidates:
        return float(base)

    lo, hi = bounds
    return float(min(min(candidates), hi) if min(candidates) > lo else lo)


def row_thresholds(
    df,
    table: dict,
    strategy: str = "pooled",
    base: float = DEFAULT_BASE,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
):
    """Vectorised per-row thresholds for a DataFrame (used by the audit)."""
    import numpy as np

    return np.array(
        [
            resolve_threshold(
                region=row["region"] if "region" in df.columns else None,
                wealth_quartile=row["wealth_quartile"] if "wealth_quartile" in df.columns else None,
                rural_flag=row["rural_flag"] if "rural_flag" in df.columns else None,
                table=table,
                strategy=strategy,
                base=base,
                bounds=bounds,
            )
            for _, row in df.iterrows()
        ],
        dtype=float,
    )
