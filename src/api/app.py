import io
import os

import pandas as pd
import yaml
from flask import Flask, jsonify, request, send_from_directory

from src.risk.explain import explain as explain_risk
from src.risk.stratifier import FEATURES, load as load_risk_model
from src.risk.thresholds import load as load_threshold_table
from src.risk.thresholds import resolve_threshold
from src.imaging.quality import assess_image
from src.routing.optimizer import min_cost_center
from src.triage.rules import decide

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
GEO_DIR = os.path.join(ROOT, "data", "geo")
RISK_PATH = os.path.join(MODELS_DIR, "risk_gb.pkl")
CENTERS_PATH = os.path.join(GEO_DIR, "centers.csv")
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
IMAGING_ONNX = os.path.join(MODELS_DIR, "imaging.onnx")
IMAGING_CLASSES = ["benign", "malignant", "normal"]
IMG_DIR = os.path.join(ROOT, "data", "images")
PATIENTS_PATH = os.path.join(ROOT, "data", "raw", "patients.csv")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

ROUTING_CFG = CFG["routing"]
RISK_CFG = CFG["risk"]
IMAGING_CFG = CFG["imaging"]
TRIAGE_CFG = CFG["triage"]

ROUTING_WEIGHTS = ROUTING_CFG["weights"]
MAX_TRAVEL_RATIO = ROUTING_CFG["max_travel_ratio"]
STRATEGY = RISK_CFG.get("threshold_strategy", "pooled")
BASE_THRESHOLD = RISK_CFG.get("risk_threshold", 0.5)
THR_BOUNDS = tuple(RISK_CFG.get("threshold_bounds", [0.02, 0.95]))
THR_TABLE_PATH = os.path.join(ROOT, RISK_CFG.get("group_thresholds_path", "config/group_thresholds.json"))
CONFIDENCE_FLOOR = IMAGING_CFG.get("confidence_floor", 0.65)
OOD_GUARD = bool(IMAGING_CFG.get("ood_guard", True))
ESCALATE_THRESHOLD = TRIAGE_CFG.get("escalate_threshold", 0.8)

# region -> OSM road network (GraphML) for travel-time routing
NETWORK_FILES = {"Bamako": "bamako_center.graphml"}

# minimum equipment a center must have to handle each triage level
REQUIRED_EQUIPMENT = {
    "wait": [],
    "local_scan": ["us"],
    "regional_imaging": ["mammo"],
    "urgent_biopsy": ["biopsy", "surgery"],
}

app = Flask(__name__, static_folder="static", static_url_path="/static")

_risk_model = None
_centers = None
_networks: dict[str, object] = {}
_imaging_session = None
_threshold_table = None
_cohort = None


def _get_risk_model():
    global _risk_model
    if _risk_model is None:
        _risk_model = load_risk_model(RISK_PATH)
    return _risk_model


def _get_centers() -> pd.DataFrame:
    global _centers
    if _centers is None:
        _centers = pd.read_csv(CENTERS_PATH)
    return _centers


def _get_cohort():
    """Cohort baselines for risk explainability; None if the file is absent."""
    global _cohort
    if _cohort is None and os.path.exists(PATIENTS_PATH):
        _cohort = pd.read_csv(PATIENTS_PATH)
    return _cohort


def _get_threshold_table() -> dict:
    global _threshold_table
    if _threshold_table is None:
        _threshold_table = load_threshold_table(THR_TABLE_PATH)
    return _threshold_table


def _get_network(region: str):
    name = NETWORK_FILES.get(region, "")
    if not name:
        return None
    if name not in _networks:
        import osmnx as ox

        _networks[name] = ox.load_graphml(os.path.join(GEO_DIR, name))
    return _networks[name]


def _get_imaging_session():
    global _imaging_session
    if _imaging_session is None:
        import onnxruntime as ort

        _imaging_session = ort.InferenceSession(IMAGING_ONNX, providers=["CPUExecutionProvider"])
    return _imaging_session


def _classify(image_bytes: bytes) -> dict:
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    if OOD_GUARD:
        quality = assess_image(image_bytes)
        if not quality["ok"]:
            return {
                "class": None,
                "confidence": 0.0,
                "above_floor": False,
                "probs": None,
                "suspicious": True,
                "quality": quality,
            }
    else:
        quality = {"ok": True, "reasons": [], "checks": {}}

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = tf(Image.open(io.BytesIO(image_bytes)).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
    probs = _get_imaging_session().run(None, {"image": img})[0][0].tolist()
    idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    conf = probs[idx]
    return {
        "class": IMAGING_CLASSES[idx],
        "confidence": conf,
        "above_floor": bool(conf >= CONFIDENCE_FLOOR),
        "probs": {c: round(p, 4) for c, p in zip(IMAGING_CLASSES, probs)},
        "suspicious": False,
        "quality": quality,
    }


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/classify")
def classify():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "missing image"}), 400
    try:
        payload = f.read()
        result = _classify(payload)
        os.makedirs(IMG_DIR, exist_ok=True)
        rel = os.path.join("data", "images", f.filename or "scan.png")
        with open(os.path.join(ROOT, rel), "wb") as out:
            out.write(payload)
        result["stored"] = rel
        return jsonify(result)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok", "risk_model": RISK_PATH, "networks": list(NETWORK_FILES)})


@app.post("/triage")
def triage():
    data = request.get_json(force=True)

    row = pd.DataFrame([{col: data.get(col, 0) for col in FEATURES}])
    risk_score = float(_get_risk_model().predict_proba(row[FEATURES])[:, 1][0])

    risk_threshold = resolve_threshold(
        region=data.get("region"),
        wealth_quartile=data.get("wealth_quartile"),
        rural_flag=data.get("rural_flag"),
        table=_get_threshold_table(),
        strategy=STRATEGY,
        base=BASE_THRESHOLD,
        bounds=THR_BOUNDS,
    )

    level = decide(
        risk_score=risk_score,
        image_result=data.get("image_result"),
        escalate_threshold=ESCALATE_THRESHOLD,
        risk_threshold=risk_threshold,
    )

    explanation = explain_risk(_get_risk_model(), row, FEATURES, cohort=_get_cohort(), top_k=4)

    region = data.get("region", "")
    centers = _get_centers()
    origin = None
    if data.get("lat") is not None and data.get("lon") is not None:
        origin = {"lat": float(data["lat"]), "lon": float(data["lon"])}
    elif region:
        match = centers[centers["name"].str.lower() == region.lower()]
        if not match.empty:
            c = match.iloc[0]
            origin = {"lat": float(c["lat"]), "lon": float(c["lon"])}

    center = None
    if origin is not None:
        required = set(REQUIRED_EQUIPMENT.get(level, []))
        candidates = []
        for _, r in centers.iterrows():
            have = set(str(r["equipment"]).split(","))
            if not required.issubset(have):
                continue
            candidates.append(
                {
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "name": r["name"],
                    "level": r["level"],
                    "out_of_pocket": float(r["base_cost"]),
                    "load": 0.5,
                }
            )
        best = min_cost_center(origin, candidates, ROUTING_WEIGHTS, _get_network(region), max_travel_ratio=MAX_TRAVEL_RATIO)
        best.pop("_cost", None)
        center = best

    return jsonify(
        {
            "risk_score": risk_score,
            "risk_threshold": risk_threshold,
            "threshold_strategy": STRATEGY,
            "explain_method": explanation["method"],
            "factors": explanation["factors"],
            "triage": level,
            "center": center,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)