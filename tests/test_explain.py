import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.explain import explain  # noqa: E402
from src.risk.stratifier import FEATURES, load  # noqa: E402

FEATURE_VALUES = {
    "age": 62,
    "family_history": 1,
    "prior_biopsy": 0,
    "palpability": 1,
    "symptom_duration_months": 12,
    "rural_flag": 1,
    "wealth_quartile": 1,
}


@pytest.fixture(scope="module")
def model():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "risk_gb.pkl")
    return load(path)


def test_explain_structure(model, cohort):
    row = pd.DataFrame([FEATURE_VALUES])
    out = explain(model, row, FEATURES, cohort=cohort, top_k=4)
    assert out["method"] in {"tree_shap", "marginal_effect"}
    assert 0.0 <= out["baseline_risk"] <= 1.0
    assert 0.0 <= out["risk_score"] <= 1.0
    assert len(out["factors"]) == 4
    signs = [abs(f["contribution"]) for f in out["factors"]]
    assert signs == sorted(signs, reverse=True)


def test_explain_matches_model_eval(model, cohort):
    row = pd.DataFrame([FEATURE_VALUES])
    expected = float(model.predict_proba(row[FEATURES])[:, 1][0])
    assert explain(model, row, FEATURES, cohort=cohort)["risk_score"] == pytest.approx(expected, abs=1e-4)


def test_tops_driver_is_symptom_duration(model, cohort):
    row = pd.DataFrame([FEATURE_VALUES])
    out = explain(model, row, FEATURES, cohort=cohort)
    assert out["factors"][0]["feature"] == "symptom_duration_months"
    assert out["factors"][0]["contribution"] > 0


def test_shap_runs_without_cohort(model):
    row = pd.DataFrame([FEATURE_VALUES])
    out = explain(model, row, FEATURES)
    assert 0.0 <= out["baseline_risk"] <= 1.0
    assert len(out["factors"]) == len(FEATURES)


def test_low_risk_patient_negative_drivers(model, cohort):
    row = pd.DataFrame([{f: 0 for f in FEATURES}])
    out = explain(model, row, FEATURES, cohort=cohort)
    assert out["risk_score"] < 0.05
    assert out["factors"][0]["contribution"] <= 0