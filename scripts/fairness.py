"""Fairness audit for the risk stratifier across wealth/rurality/region groups.

Reports prevalence, group AUC, TPR/FPR at the triage threshold, equalized-odds
disparities, demographic parity gap, and calibration (ECE) per group.
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


def report_group(df: pd.DataFrame, mask: np.ndarray, y: np.ndarray, scores: np.ndarray) -> dict:
    s, t = scores[mask], y[mask]
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    pred = s >= THRESHOLD
    return {
        "n": n,
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


def main() -> None:
    df = pd.read_csv(os.path.join(ROOT, "data", "raw", "patients.csv"))
    model = load(os.path.join(ROOT, "models", "risk_gb.pkl"))
    scores = model.predict_proba(df[FEATURES])[:, 1]
    y = df["label"].to_numpy()

    audit = {"threshold": THRESHOLD, "overall": report_group(df, np.ones(len(df), bool), y, scores), "groups": {}}

    groups = {
        "wealth_quartile": sorted(df["wealth_quartile"].unique()),
        "rural_flag": sorted(df["rural_flag"].unique()),
        "region": sorted(df["region"].unique()),
    }
    for col, values in groups.items():
        audit["groups"][col] = {}
        for v in values:
            mask = (df[col] == v).to_numpy()
            audit["groups"][col][str(v)] = report_group(df, mask, y, scores)

    disparities = {}
    for col in ["wealth_quartile", "rural_flag", "region"]:
        g = audits_tpr_fprs(audit, col)
        disparities[col] = {
            "max_tpr_diff": max(x["tpr"] for x in g) - min(x["tpr"] for x in g),
            "max_fpr_diff": max(x["fpr"] for x in g) - min(x["fpr"] for x in g),
            "selection_rate_max_diff": max(x["selection_rate"] for x in g) - min(x["selection_rate"] for x in g),
            "auc_max_diff": max(x["auc"] for x in g) - min(x["auc"] for x in g),
        }
    audit["disparities"] = disparities

    out = os.path.join(ROOT, "data", "processed", "fairness_report.json")
    with open(out, "w") as f:
        json.dump(audit, f, indent=2, default=float)
    print(f"saved -> {out}\n")

    for col in ["wealth_quartile", "rural_flag", "region"]:
        print(f"== {col} ==")
        print("grp   n   prev  auc    sel  tpr   fpr   ece")
        for k, v in audit["groups"][col].items():
            if v.get("n") == 0:
                continue
            def fmt(x, d=3):
                return "-" if x is None else f"{x:.{d}f}"
            print(f"{k:<4} {v['n']:4d} {fmt(v['prevalence'])}  {fmt(v['auc'])}  {fmt(v['selection_rate'])}  {fmt(v['tpr'])}  {fmt(v['fpr'])}  {fmt(v['ece'])}")
        d = disparities[col]
        print(f"   disparities: TPR {d['max_tpr_diff']:.2f} | FPR {d['max_fpr_diff']:.2f} | sel-rate {d['selection_rate_max_diff']:.2f} | AUC {d['auc_max_diff']:.2f}\n")

    print(f"overall AUC {audit['overall']['auc']:.3f} | ECE {audit['overall']['ece']:.3f} | sel-rate {audit['overall']['selection_rate']:.3f}")


def audits_tpr_fprs(audit: dict, col: str) -> list[dict]:
    return [v for v in audit["groups"][col].values() if v.get("n") > 0]


if __name__ == "__main__":
    main()