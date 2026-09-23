import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import IMAGING_CLASSES  # noqa: E402


def test_health(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_classify_real_image(app_client, real_image):
    r = app_client.post(
        "/classify",
        data={"image": (io.BytesIO(real_image), "scan.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["suspicious"] is False
    assert body["class"] in IMAGING_CLASSES
    assert body["quality"]["ok"] is True
    assert 0.0 <= body["confidence"] <= 1.0


def test_classify_rejects_ood(app_client, white_image):
    r = app_client.post(
        "/classify",
        data={"image": (io.BytesIO(white_image), "white.png")},
        content_type="multipart/form-data",
    )
    body = r.get_json()
    assert body["suspicious"] is True
    assert body["class"] is None
    assert body["quality"]["ok"] is False


def test_classify_missing_file(app_client):
    assert app_client.post("/classify", data={}).status_code == 400


def test_triage_group_threshold_and_factors(app_client):
    payload = {
        "age": 62,
        "family_history": 1,
        "prior_biopsy": 0,
        "palpability": 1,
        "symptom_duration_months": 12,
        "rural_flag": 1,
        "wealth_quartile": 1,
        "region": "Koulikoro",
    }
    r = app_client.post("/triage", json=payload)
    assert r.status_code == 200
    body = r.get_json()
    assert body["risk_score"] > 0.5
    assert body["risk_threshold"] == pytest.approx(0.37)  # Koulikoro group rule
    assert body["threshold_strategy"] == "group"
    assert body["explain_method"] == "tree_shap"
    assert body["factors"] and body["factors"][0]["feature"] == "symptom_duration_months"
    assert body["triage"] in {"wait", "local_scan", "regional_imaging", "urgent_biopsy"}
    assert body["center"] is not None


def test_triage_low_risk_wait(app_client):
    payload = {
        "age": 40,
        "family_history": 0,
        "prior_biopsy": 0,
        "palpability": 0,
        "symptom_duration_months": 1,
        "rural_flag": 0,
        "wealth_quartile": 4,
        "region": "Bamako",
    }
    body = app_client.post("/triage", json=payload).get_json()
    assert body["risk_threshold"] == pytest.approx(0.25)  # Bamako group rule
    assert body["triage"] == "wait"


def test_triage_unknown_region_falls_back_to_pooled(app_client):
    payload = {
        "age": 55,
        "family_history": 0,
        "prior_biopsy": 0,
        "palpability": 0,
        "symptom_duration_months": 3,
        "rural_flag": 0,
        "wealth_quartile": 9,  # not a valid quartile -> no group entry -> pooled fallback
        "region": "Nowhere",
    }
    body = app_client.post("/triage", json=payload).get_json()
    assert body["risk_threshold"] == pytest.approx(0.5)


def test_triage_unknown_region_but_known_wealth_uses_wealth(app_client):
    payload = {
        "age": 55,
        "family_history": 0,
        "prior_biopsy": 0,
        "palpability": 0,
        "symptom_duration_months": 3,
        "rural_flag": 0,
        "wealth_quartile": 2,
        "region": "Nowhere",
    }
    body = app_client.post("/triage", json=payload).get_json()
    assert body["risk_threshold"] == pytest.approx(0.48)


def test_triage_with_image_escalation(app_client, busi_image_path):
    with open(busi_image_path, "rb") as f:
        up = app_client.post(
            "/classify",
            data={"image": (io.BytesIO(f.read()), "scan.png")},
            content_type="multipart/form-data",
        ).get_json()
    payload = {
        "age": 55,
        "family_history": 0,
        "prior_biopsy": 0,
        "palpability": 1,
        "symptom_duration_months": 6,
        "rural_flag": 0,
        "wealth_quartile": 2,
        "region": "Bamako",
        "image_result": {"confidence": up["confidence"], "above_floor": up["above_floor"]},
    }
    body = app_client.post("/triage", json=payload).get_json()
    assert body["triage"] in {"local_scan", "regional_imaging", "urgent_biopsy", "wait"}