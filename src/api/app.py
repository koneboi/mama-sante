import io
import os

import pandas as pd
import yaml
from flask import Flask, jsonify, request, send_from_directory

from src.risk.stratifier import FEATURES, load as load_risk_model
from src.routing.optimizer import min_cost_center
from src.triage.rules import decide

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
GEO_DIR = os.path.join(ROOT, "data", "geo")
RISK_PATH = os.path.join(MODELS_DIR, "risk_gb.pkl")
CENTERS_PATH = os.path.join(GEO_DIR, "centers.csv")
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
IMAGING_ONNX = os.path.join(MODELS_DIR, "imaging.onnx")
IMAGING_META = os.path.join(MODELS_DIR, "imaging.json")
IMAGING_CLASSES = ["benign", "malignant", "normal"]
IMG_DIR = os.path.join(ROOT, "data", "images")

with open(CONFIG_PATH) as f:
    ROUTING_CFG = yaml.safe_load(f)["routing"]

ROUTING_WEIGHTS = ROUTING_CFG["weights"]
MAX_TRAVEL_RATIO = ROUTING_CFG["max_travel_ratio"]

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
_CONFIDENCE_FLOOR = 0.65


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
        "above_floor": bool(conf >= _CONFIDENCE_FLOOR),
        "probs": {c: round(p, 4) for c, p in zip(IMAGING_CLASSES, probs)},
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
        result = _classify(f.read())
        os.makedirs(IMG_DIR, exist_ok=True)
        f.seek(0)
        rel = os.path.join("data", "images", f.filename or "scan.png")
        with open(os.path.join(ROOT, rel), "wb") as out:
            out.write(f.read())
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

    level = decide(
        risk_score=risk_score,
        image_result=data.get("image_result"),
        escalate_threshold=0.8,
    )

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

    return jsonify({"risk_score": risk_score, "triage": level, "center": center})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)