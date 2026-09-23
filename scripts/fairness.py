"""Fairness audit for the risk stratifier across wealth/rurality/region groups.

Reports prevalence, group AUC, TPR/FPR at the triage threshold, equalized-odds
disparities, demographic parity gap, and calibration (ECE) per group.

Also calibrates group-aware decision thresholds (fairness by design):
for every audited group a threshold is chosen so its TPR lands as close as
possible to the pooled TPR (ties broken by FPR near the pooled FPR, then by
proximity to the base threshold). The deployed rule is per-row
``min(region_threshold, wealth_threshold)``; the audit re-scores every group
under that rule and reports the before/after disparities. The resulting table
is written to the committed ``config/group_thresholds.json`` for the API.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.risk.stratifier import FEATURES, load  # noqa: E402

THRESHOLD = 0.5
N_BINS = 10
BOUNDS = (0.02, 0.95)
CALIB_STEP = 0.01


def ece(scores: np.ndarray, y: np.ndarray) -> float:
    edges = np.linspace(0, 1, N_BINS + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & (scores < hi)
        if m.sum() == 0:
            continue
        conf, acc = scores[m].mean(), y[m].mean()
        total += m.sum() * abs(conf - acc)
    n = len(scores) or 1
    return total / n


def report_group(
    df: pd.DataFrame,
    mask: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    row_t: np.ndarray | None = None,
) -> dict:
    s, t = scores[mask], y[mask]
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    if row_t is None:
        thr, pred = THRESHOLD, s >= THRESHOLD
    else:
        thr = float(np.median(row_t[mask]))
        pred = s >= row_t[mask]  # per-row thresholds (min of region + wealth)
    return {
        "n": n,
        "threshold": thr,
        "prevalence": float(t.mean()),
        "mean_risk_score": float(s.mean()),
        "auc": float(roc_auc(t, s)),
        "selection_rate": float(pred.mean()),
        "tpr": float((pred & (t == 1)).sum() / max((t == 1).sum(), 1)),
        "fpr": float((pred & (t == 0)).sum() / max((t == 0).sum(), 1)),
        "ece": float(ece(s, t)),
        "calib_pos": float(s[t == 1].mean()) if (t == 1).any() else None,
        "calib_neg": float(s[t == 0].mean()) if (t == 0).any() else None,
    }


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, s) if y.sum() and (y == 0).sum() else float("nan")


def calibrate_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    target_tpr: float,
    target_fpr: float,
    base: float = THRESHOLD,
) -> float | None:
    """Threshold whose TPR is closest to the pooled TPR for this group."""
    s, t = scores[mask], y[mask]
    n_pos, n_neg = int((t == 1).sum()), int((t == 0).sum())
    if n_pos == 0:
        return None

    best_t, best_key = None, None
    for t_try in np.arange(BOUNDS[0], BOUNDS[1] + 1e-9, CALIB_STEP):
        pred = s >= t_try
        tpr = (pred & (t == 1)).sum() / n_pos
        fpr = (pred & (t == 0)).sum() / max(n_neg, 1)
        # closest TPR first, then FPR near pooled FPR, then closest to base
        key = (round(abs(tpr - target_tpr), 6), round(abs(fpr - target_fpr), 6), round(abs(t_try - base), 6))
        if best_key is None or key < best_key:
            best_t, best_key = float(t_try), key
    return best_t


def build_group_table(
    df: pd.DataFrame,
    y: np.ndarray,
    scores: np.ndarray,
    target_tpr: float,
    target_fpr: float,
) -> dict:
    table: dict = {
        "base": THRESHOLD,
        "target_tpr": target_tpr,
        "bounds": list(BOUNDS),
        "region": {},
        "wealth_quartile": {},
    }
    # Note: only region + wealth are calibrated into the combined
    # min(region_t, wealth_t) rule. rural_flag stays audited and reported:
    # coupling it into the min-rule pulls thresholds of already-correct groups
    # and widens the (small) region residual, so it is deliberately reported-only.
    for col in ("region", "wealth_quartile"):
        for v in sorted(df[col].unique()):
            mask = (df[col] == v).to_numpy()
            t = calibrate_threshold(y, scores, mask, target_tpr, target_fpr)
            if t is not None:
                table[col][str(v)] = round(t, 3)
    return table


def apply_group_thresholds(df: pd.DataFrame, table: dict) -> np.ndarray:
    from src.risk.thresholds import row_thresholds

    return row_thresholds(df, table, strategy="group", base=table.get("base", THRESHOLD), bounds=tuple(table.get("bounds", BOUNDS)))


def disparity_summary(audit: dict) -> dict:
    out = {}
    for col in ["wealth_quartile", "rural_flag", "region"]:
        g = audits_tpr_fprs(audit, col)
        out[col] = {
            "max_tpr_diff": max(x["tpr"] for x in g) - min(x["tpr"] for x in g),
            "max_fpr_diff": max(x["fpr"] for x in g) - min(x["fpr"] for x in g),
            "selection_rate_max_diff": max(x["selection_rate"] for x in g) - min(x["selection_rate"] for x in g),
            "auc_max_diff": max(x["auc"] for x in g) - min(x["auc"] for x in g),
        }
    return out


def build_audit(df, y, scores, row_t=None) -> dict:
    audit = {
        "threshold": THRESHOLD if row_t is None else "group",
        "overall": report_group(df, np.ones(len(df), bool), y, scores, row_t),
        "groups": {},
    }
    groups = {
        "wealth_quartile": sorted(df["wealth_quartile"].unique()),
        "rural_flag": sorted(df["rural_flag"].unique()),
        "region": sorted(df["region"].unique()),
    }
    for col, values in groups.items():
        audit["groups"][col] = {}
        for v in values:
            mask = (df[col] == v).to_numpy()
            audit["groups"][col][str(v)] = report_group(df, mask, y, scores, row_t)
    audit["disparities"] = disparity_summary(audit)
    return audit


def main() -> None:
    df = pd.read_csv(os.path.join(ROOT, "data", "raw", "patients.csv"))
    model = load(os.path.join(ROOT, "models", "risk_gb.pkl"))
    scores = model.predict_proba(df[FEATURES])[:, 1]
    y = df["label"].to_numpy()

    # ---- pooled audit (baseline) ----
    audit = build_audit(df, y, scores)

    # ---- fairness by design: calibrate group thresholds ----
    pooled = audit["overall"]
    table = build_group_table(df, y, scores, target_tpr=pooled["tpr"], target_fpr=pooled["fpr"])
    table["strategy"] = "group"
    row_t = apply_group_thresholds(df, table)
    after = build_audit(df, y, scores, row_t=row_t)

    improvement = {
        col: {
            "before": audit["disparities"][col]["max_tpr_diff"],
            "after": after["disparities"][col]["max_tpr_diff"],
        }
        for col in ("wealth_quartile", "rural_flag", "region")
    }

    report = dict(audit)  # pooled keys stay at top level (back-compat)
    report["group_thresholds"] = table
    report["after_group_thresholds"] = after
    report["tpr_disparity_improvement"] = improvement

    out = os.path.join(ROOT, "data", "processed", "fairness_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"saved -> {out}\n")

    # canonical, committed threshold table (data/processed is git-ignored)
    thr_path = os.path.join(ROOT, "config", "group_thresholds.json")
    os.makedirs(os.path.dirname(thr_path), exist_ok=True)
    with open(thr_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"saved -> {thr_path}\n")

    for col in ["wealth_quartile", "rural_flag", "region"]:
        print(f"== {col} ==")
        print("grp   n   prev  auc    sel  tpr   fpr   ece   thr")
        for k, v in audit["groups"][col].items():
            if v.get("n") == 0:
                continue
            a2 = after["groups"][col][k]

            def fmt(x, d=3):
                return "-" if x is None else f"{x:.{d}f}"
            print(
                f"{k:<4} {v['n']:4d} {fmt(v['prevalence'])}  {fmt(v['auc'])}  {fmt(v['selection_rate'])}  "
                f"{fmt(v['tpr'])}  {fmt(v['fpr'])}  {fmt(v['ece'])}  {fmt(a2.get('threshold'))}"
            )
        d = audit["disparities"][col]
        d2 = after["disparities"][col]
        print(
            f"   disparities TPR: {d['max_tpr_diff']:.3f} -> {d2['max_tpr_diff']:.3f} "
            f"| FPR: {d['max_fpr_diff']:.3f} -> {d2['max_fpr_diff']:.3f} | sel-rate: "
            f"{d['selection_rate_max_diff']:.3f} -> {d2['selection_rate_max_diff']:.3f}\n"
        )

    print(f"overall AUC {pooled['auc']:.3f} | ECE {pooled['ece']:.3f} | pooled TPR {pooled['tpr']:.3f}")
    print(
        "group-threshold rule overall: "
        f"TPR {after['overall']['tpr']:.3f} | FPR {after['overall']['fpr']:.3f} | "
        f"sel-rate {after['overall']['selection_rate']:.3f}"
    )
    print(f"{len(table['region'])} region + {len(table['wealth_quartile'])} wealth thresholds calibrated "
          f"(rural_flag audited only: min-rule coupling widens the region residual)")


def audits_tpr_fprs(audit: dict, col: str) -> list[dict]:
    return [v for v in audit["groups"][col].values() if v.get("n") > 0]


if __name__ == "__main__":
    main()
