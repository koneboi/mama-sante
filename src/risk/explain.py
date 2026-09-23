"""Explainability for the risk stratifier.

Primary method: exact SHAP tree-shap values (``shap.TreeExplainer``).
Fallback (e.g. when shap is not installed): marginal-effect attribution that
moves each feature one at a time from a cohort baseline to the query value and
records the resulting change in predicted risk. Both are per-prediction, local
attributions; the fallback is approximate (ignores interactions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap as _shap  # type: ignore

    _HAVE_SHAP = True
except Exception:  # pragma: no cover - depends on environment
    _HAVE_SHAP = False


def baseline_row(cohort: pd.DataFrame | None, feature_cols: list[str]) -> pd.DataFrame:
    if cohort is not None and len(cohort):
        return pd.DataFrame([{f: float(cohort[f].median()) for f in feature_cols}])
    return pd.DataFrame([{f: 0.0 for f in feature_cols}])


def _shap_explain(model, row: pd.DataFrame, feature_cols: list[str]):
    """Return (expected_risk, {feature: contribution}) or None."""
    if not _HAVE_SHAP:
        return None
    try:
        explainer = _shap.TreeExplainer(model)
        sv = explainer.shap_values(row[feature_cols])
        if isinstance(sv, list):  # legacy binary output: list of per-class arrays
            sv = sv[1] if len(sv) == 2 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 2:
            sv = sv[0]
        exp = explainer.expected_value
        if isinstance(exp, (list, np.ndarray)):
            exp = exp[-1]
        expected = float(exp)
        return expected, {f: float(sv[i]) for i, f in enumerate(feature_cols)}
    except Exception:
        return None


def _marginal_explain(model, row: pd.DataFrame, feature_cols: list[str], base: pd.DataFrame):
    base_score = float(model.predict_proba(base[feature_cols])[:, 1][0])
    contribs: dict[str, float] = {}
    for f in feature_cols:
        probe = base.copy()
        probe[f] = float(row[f].iloc[0])
        contribs[f] = float(model.predict_proba(probe[feature_cols])[:, 1][0]) - base_score
    return base_score, contribs


def explain(
    model,
    row,
    feature_cols: list[str],
    cohort: pd.DataFrame | None = None,
    top_k: int | None = None,
) -> dict:
    """Local attribution for one patient row. Row may be a Series or 1-row DataFrame."""
    if isinstance(row, pd.DataFrame):
        row = row.iloc[[0]]
    else:
        row = pd.DataFrame([row.to_dict()])

    base = baseline_row(cohort, feature_cols)
    risk_score = float(model.predict_proba(row[feature_cols])[:, 1][0])
    baseline_risk = float(model.predict_proba(base[feature_cols])[:, 1][0])

    result = _shap_explain(model, row, feature_cols)
    if result is not None:
        _expected, contribs = result
        method = "tree_shap"
    else:
        _expected, contribs = _marginal_explain(model, row, feature_cols, base)
        method = "marginal_effect"

    factors = sorted(
        (
            {"feature": f, "value": float(row[f].iloc[0]), "baseline": float(base[f].iloc[0]),
             "contribution": round(c, 5)}
            for f, c in contribs.items()
        ),
        key=lambda x: abs(x["contribution"]),
        reverse=True,
    )
    if top_k:
        factors = factors[:top_k]

    return {
        "method": method,
        "baseline_risk": round(baseline_risk, 5),
        "risk_score": round(risk_score, 5),
        "factors": factors,
    }