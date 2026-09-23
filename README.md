# mama-santé

**Computational breast-cancer care platform for Mali** — risk stratification, ultrasound image classification, referral triage, and geographic routing in one web application, built to attack the three barriers named in the original problem note: **distance, money, and time**.

![Risk model](https://img.shields.io/badge/Risk%20stratification-GB%20CV%20AUC%200.733-e11d48?style=flat-square)
![Imaging](https://img.shields.io/badge/Imaging-MobileNetV3%20val%200.940-0d6efd?style=flat-square)
![Int8 on-device](https://img.shields.io/badge/On--device-int8%20TFLite%201.2%20MB%20%C2%B7%200.803-16a34a?style=flat-square)
![Routing](https://img.shields.io/badge/Routing-OSM%20Bamako%2088k%20edges-7c3aed?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)

---

## 1. The problem

From `note.md`, verbatim:

> the problem of breast cancer treatment in Mali are: **distance, money, and time**. How to fix these ?

Late presentation and long travel to scarce diagnostic centres mean patients arrive
advanced, costs of repeated trips bankrupt families, and referrals sit in queues while
cancers progress. This project builds the software layer that attacks all three:

| Barrier | What this platform does |
|---|---|
| **Distance** | Network-based routing to the *nearest capable* centre (real OSM roads of Bamako + regional fallbacks), so patients travel less and to a centre that can actually do the required test |
| **Money** | Out-of-pocket cost is an explicit routing objective (transport at 150 FCFA/km); capability filtering prevents wasted trips to centres that lack the equipment |
| **Time** | Risk stratification + 4-level triage escalate the urgent cases first; time is the dominant routing objective (weight 40.0 vs 0.2 for km) |

---

## 2. What it does

Four coupled modules behind one Flask API and a mobile-first web UI:

1. **Risk stratification** (`src/risk/`) — GradientBoosting over 7 clinical/demographic
   features (age, family history, prior biopsy, palpability, symptom duration, rural
   flag, wealth quartile) → risk score.
2. **Ultrasound image classification** (`src/imaging/`) — MobileNetV3-Small →
   benign / malignant / normal from BUSI ultrasound images; served through ONNX Runtime.
3. **Referral triage** (`src/triage/rules.py`) — deterministic 4-level decision:
   `wait → local_scan → regional_imaging → urgent_biopsy` combining the risk score,
   imaging result (confidence floor 0.65) and escalation threshold 0.8.
4. **Geographic routing** (`src/routing/`) — picks the referral centre by minimising a
   weighted cost over travel time, distance, out-of-pocket cost and centre load, subject
   to a **capability filter** (the centre must hold the equipment the triage level needs)
   and a **time-fairness guard** (no centre > 1.5× the travel time of the fastest option).

The web UI (`src/api/static/`) lets a health worker enter the patient, upload an
ultrasound image, and get back: risk gauge → triage badge → recommended centre with
travel time, cost, and an OpenStreetMap directions link.

---

## 3. Repository structure

```
mama-sante/
├── config.yaml                  # all tunables: features, thresholds, routing weights
├── note.md                      # original problem statement
├── requirements.txt
├── src/
│   ├── risk/stratifier.py       # GradientBoosting train/score/save/load
│   ├── imaging/classifier.py    # MobileNetV3-Small build/predict (torch + ONNX)
│   ├── triage/rules.py          # 4-level decide()
│   ├── routing/optimizer.py     # travel_metrics, min_cost_center (time-first)
│   └── api/
│       ├── app.py               # Flask: / /health /classify /triage
│       └── static/              # index.html, styles.css, app.js (mobile UI)
├── scripts/
│   ├── generate_synthetic_data.py   # 2000 patients, 12 Malian centres
│   ├── train_risk.py                # risk GB + CV AUC
│   ├── train_imaging.py             # torch MobileNetV3 + 15% holdout
│   ├── export_imaging.py            # fp32 + fp16 ONNX, parity check
│   ├── quantize_imaging.py          # ORT int8 PTQ probes (see findings F1)
│   ├── train_tflite_imaging.py      # Keras train → fp32/int8 TFLite (phone artifact)
│   ├── train_qat_imaging.py         # torch eager QAT reference (see findings F3)
│   ├── fetch_network.py             # OSM Bamako extract + per-edge travel time
│   └── fairness.py                  # subgroup audit → fairness_report.json
├── notebooks/EDA.ipynb             # exploratory analysis, baseline CV
├── data/
│   ├── raw/        patients.csv (generated)          [git-ignored]
│   ├── processed/  busi_class/ 780 images, fairness  [git-ignored]
│   └── geo/        centres.csv (kept), bamako graphml [31 MB, git-ignored]
├── models/         imaging.json kept; weights git-ignored (see §7)
└── tests/
```

---

## 4. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. synthetic cohort + centres
python scripts/generate_synthetic_data.py

# 2. risk model                     -> models/risk_gb.pkl
python scripts/train_risk.py

# 3. imaging model + exports        -> models/imaging.{pt,onnx} + fp16
python scripts/train_imaging.py
python scripts/export_imaging.py

# 4. phone artifacts (full int8)    -> models/imaging_{fp32,int8}.tflite
python scripts/train_tflite_imaging.py

# 5. road network for routing       -> data/geo/bamako_center.graphml
python scripts/fetch_network.py

# 6. fairness audit                 -> data/processed/fairness_report.json
python scripts/fairness.py

# 7. run the app  -> http://localhost:5000
python -m src.api.app
```

Data note: BUSI was taken from the HuggingFace mirror
[`gymprathap/Breast-Cancer-Ultrasound-Images-Dataset`](https://huggingface.co/datasets/gymprathap/Breast-Cancer-Ultrasound-Images-Dataset)
(Kaggle's direct download was too slow); masks were stripped — including the
`*_mask_N.png` variants — leaving **780 images: 437 benign / 210 malignant / 133 normal**.

---

## 5. Full build log — every step from the beginning

### Step 0 — Problem framing
Read `note.md`, translated the three barriers (distance, money, time) into four
computable problems: *who is high-risk* (risk), *what does the scan show* (imaging),
*what level of care is needed* (triage), *where should the patient go* (routing).

### Step 1 — Scaffold & configuration
Project skeleton created directly in the working directory. Everything tunable lives in
`config.yaml` (feature list, thresholds, routing weights, imaging settings) so the app
never hard-codes policy. Environment: Python 3.10.18, CUDA available, torch 2.8.0,
tensorflow-cpu 2.19.0, sklearn 1.7.2, flask 3.0.3, osmnx 2.0.7, onnx/onnxruntime 1.23.

### Step 2 — Data
- **Synthetic cohort**: `generate_synthetic_data.py` → 2,000 patient records with
  realistic correlations (age, rural flag, wealth quartile, symptom duration…) plus a
  12-centre Malian facility table (`data/geo/centers.csv`: post → district → regional →
  hospital levels, equipment inventory, load).
- **Real imaging data**: BUSI ultrasound set cleaned to 780 images (masks removed).
- **EDA** (`notebooks/EDA.ipynb`, executed): class balance inspected, a baseline
  logistic model established **CV AUC 0.773** as the number to beat/understand.

### Step 3 — Risk stratification
`train_risk.py`: GradientBoosting on the 7 configured features, seeded CV.
**Result: CV AUC 0.733** (`models/risk_gb.pkl`). The score feeds `/triage`.

### Step 4 — Imaging model
`train_imaging.py`: torchvision MobileNetV3-Small, ImageNet weights, 15% holdout split
(seed 0 → 663 train / 117 val), ImageNet normalisation, flip/crop augmentation,
weighted sampler against class imbalance.
**Result: best val accuracy 0.940** (`models/imaging.pt`).

### Step 5 — ONNX export & serving format
`export_imaging.py`: traced export with softmax baked into the graph, dynamic batch axis.
- fp32 `models/imaging.onnx` (6.1 MB) — **parity vs torch 1.22e-06**, CPU latency
  ~23–32 ms/image.
- fp16 `models/imaging_fp16.onnx` (3.1 MB) — lightweight GPU/NPU copy.
> ⚠️ Finding **F4**: in torch 2.8 `torch.onnx.export` *mutates the traced model
> in-memory; always reload a fresh model before running parity checks.*

### Step 6 — Quantization for phones (three routes investigated, see §6)
ORT PTQ (failed — F1), TFLite full-int8 (succeeded — the phone artifact), torch QAT
(training succeeded, export blocked — F3). Final artifacts in §7.

### Step 7 — Triage engine
`rules.py`: pure-function `decide()` over risk score, imaging result and confidence
floor 0.65, returning one of the four configured levels; escalation threshold 0.8.

### Step 8 — Geographic routing
- `fetch_network.py` pulled the real OpenStreetMap extract of Bamako:
  **30,232 nodes / 88,257 edges**, per-edge `travel_time` added via `add_travel_time`
  (31 MB graphml, git-ignored; fetchable with one command).
- `optimizer.py`: `travel_metrics()` (network path → km, hours, FCFA, snap-guarded
  against >2 km node mismatch, haversine ×1.3 fallback when off-network) and
  `min_cost_center()` — **time-first** selection: cost =
  `0.2·km + 40·hours + 0.05·FCFA + 0.02·load`, candidates pruned to ≤ 1.5× the fastest
  travel time, then capability-filtered per triage level
  (e.g. `urgent_biopsy` requires a centre with biopsy/mammography capability).
- Transport cost model: 150 FCFA/km.

### Step 9 — API
Flask app with `/`, `/health`, `/classify` (multipart image upload → persisted under
`data/images/` → ONNX inference) and `/triage` (JSON → risk + level + centre
recommendation). Weights/thresholds read from `config.yaml`.
**Live smoke test over real HTTP**:
- `GET /` → 200 (serves the UI)
- real BUSI **malignant** image → `malignant 0.997`; benign → `benign 0.987`
- high-risk patient + image flag → `urgent_biopsy → Bamako` (2.8 km for a
  `local_scan` case; `wait` cases route to Segou/Timbuktu as expected)

### Step 10 — Mobile-first UI
`static/index.html + styles.css + app.js`: single-page patient form + image upload →
`/classify` → `/triage`, rendering risk gauge, triage badge, and centre card
(travel time, cost, OSM directions link).

### Step 11 — Fairness audit
`fairness.py` → `data/processed/fairness_report.json` on the synthetic cohort:
overall **AUC 0.896, ECE 0.024**, with subgroup gaps that matter operationally —
**region TPR disparity 0.60** and **wealth-quartile TPR disparity 0.18** (poorest
quartile under-detected). These are flagged as pre-deployment blockers: threshold
calibration per region and re-weighting of under-represented groups are required before
any real deployment.

### Step 12 — Publication
Repo pushed to GitHub; project added to the Projects menu of
[koneboi.github.io](https://koneboi.github.io/projects.html) with a detailed technical
page.

---

## 6. Key findings (most important — read this before changing anything)

### F1 — Post-training int8 quantization *collapses* on this domain (root-caused)
`scripts/quantize_imaging.py` measured ORT static/dynamic int8 against the 15% holdout:

| Recipe | val accuracy |
|---|---|
| fp32 (reference) | **0.940** |
| ORT dynamic int8 | ~0.34 |
| ORT static, MinMax | 0.342 |
| ORT static, Percentile(99) | 0.521 |
| ORT static, Entropy | 0.350 |

**Root cause**: early-conv activations on OOD ultrasound images carry extreme
outliers — **max ≈ 198 while p99 ≈ 6** — so every range-based calibration scale is
dominated by a handful of spikes and the bulk of the signal is crushed into a few
int8 bins. A control MLP quantised through the same ORT codepath stayed accurate,
proving the ORT int8 kernels themselves are fine; it is the *statistics of this
domain*. **Consequence**: PTQ is not viable here; use QAT (F3) or the TFLite path (F2).

### F2 — TFLite full-int8 works and is the shipped phone artifact
`scripts/train_tflite_imaging.py` retrains MobileNetV3-Small in Keras and converts:
**fp32 0.872 val (3.7 MB) → full-int8 0.803 val (1.21 MB)**, true int8 in *and* out
(input scale 1/127.5, zero-point −1, i.e. the exact [−1,1] range). Calibration set
size 96→300 changed nothing (0.803) — the residual gap is activation-range mismatch,
not coverage. Two environment quirks were found on the way:
- **F2a — double preprocessing bug (silent 0.538 plateau)**: Keras
  `MobileNetV3Small` ships `include_preprocessing=True` (an internal Rescaling
  1/127.5 offset −1 expecting [0,255] inputs). Externally scaling inputs to [−1,1] as
  well compressed everything into a ~0.016-wide near-constant range: the model
  *memorised* training images (train acc 0.92) but predicted **all-benign at
  confidence 1.000** on validation — reproducibly **exactly 0.538** (the benign share
  of the holdout) across runs with different external scalings, which is what
  exposed it. Fix: `include_preprocessing=False` + keep the [−1,1] scaling
  (also the cleanest semantics for int8 input quantisation). **Symptom to remember:
  an identical val number across differently-configured runs means eval is measuring
  a constant, not the model.**
- **F2b — this environment's TF/TFLite quirks**: the int8 representative dataset
  generator must `yield [batch]` (lists — a 1-tuple crashes the calibrator with
  `IndexError`); the XNNPACK int8 delegate fails to initialise, so validation must
  instantiate the interpreter with `OpResolverType.BUILTIN_REF`; keras `.h5`
  round-trip reload is broken (protobuf `MessageFactory` desync + positional-arg
  float error) — the `model.export()` SavedModel directory is the reliable source for
  conversion; the `AttributeError: 'MessageFactory'…GetPrototype` line at startup is
  a benign warning, not a crash.

### F3 — Torch eager QAT learns accurate int8, but cannot export it to ONNX
`scripts/train_qat_imaging.py`: QAT (per-tensor symmetric weight qconfig —
per-channel raises `Unsupported qscheme: per_channel_affine` at export time) trains to
**val 0.915**, i.e. quantization *is* recoverable with learned ranges. Blocked on:
- conversion must run on **CPU** (quantised cuDNN conv rejects depthwise
  `groups=16`), and
- `torch.onnx.export` cannot emit the converted graph at **any** of opsets 13/17/19 —
  `quantized::batch_norm2d` is unsupported;
- forcing BN-fold-into-conv before QAT (to remove BN from the graph) destabilised
  training on this small dataset (val never recovered past ~0.67 before the run died).
Also noted: `torch.ao.quantization.freeze_bn_stats` no longer exists in torch 2.8
(quantization API is deprecated → torchao/pt2e). **Takeaway**: for int8 on-device,
use the TFLite path (F2); keep QAT as the production recipe in a TF-based build.

### F4 — Exporter/model-handle gotcha
See Step 5: `torch.onnx.export` mutates the traced module in torch 2.8 — parity tests
must build or reload a *fresh* model first, otherwise you silently compare the model
against itself.

### F5 — Routing policy actually works on real geography
With the real Bamako graph, `local_scan` cases resolve to the 2.8 km Bamako centre,
`urgent_biopsy` escalates to the capable regional hospital, and `wait` cases in the
north fall through to Segou/Timbuktu — i.e. the capability filter + time-first cost +
1.5× fairness guard behave as designed rather than always picking the geographic
nearest facility. Time dominance (weight 40 vs 0.2 for km) was chosen deliberately:
for cancer pathways, hours matter more than distance.

### F6 — Fairness numbers that must not be shipped around
Report in `data/processed/fairness_report.json`. Headline: AUC 0.896 / ECE 0.024
looks deployment-ready *until* subgroup slicing: region TPR gap **0.60**, wealth
quartile TPR gap **0.18** — the poorest and some rural regions are systematically
under-flagged by a threshold calibrated on the pooled population.

### F7 — Operational/robustness notes
- Background `nohup` training jobs were silently killed mid-run several times;
  long trainings are safer in the foreground with a generous timeout.
- Keep `models/` honest: broken/experimental outputs were deleted rather than left
  to be mistaken for verified artifacts; `models/imaging.json` is the manifest that
  records which artifact is verified, its accuracy and its intended use.

---

## 7. Verified model artifacts

Weights are **not** committed (see `.gitignore`) — rebuild with §4, or take them from
a release. The manifest `models/imaging.json` records the same table:

| Artifact | Format | Size | Accuracy | Use |
|---|---|---|---|---|
| `models/risk_gb.pkl` | sklearn GB | 140 KB | CV AUC 0.733 | risk score |
| `models/imaging.onnx` | ONNX fp32 | 6.1 MB | val 0.940 | server `/classify` (default) |
| `models/imaging_fp16.onnx` | ONNX fp16 | 3.1 MB | parity-verified | light GPU/NPU |
| `models/imaging_fp32.tflite` | TFLite float | 3.7 MB | val 0.872 | mobile float reference |
| `models/imaging_int8.tflite` | TFLite **full-int8** | **1.21 MB** | **val 0.803** | **on-device phone** |

Holdout = 15% seeded split (seed 0, 663/117) for all imaging numbers.

---

## 8. API

| Method | Route | Body | Returns |
|---|---|---|---|
| GET | `/` | — | mobile UI |
| GET | `/health` | — | service + model status |
| POST | `/classify` | `multipart/form-data: image` | `{class, confidence, probs, above_floor}` |
| POST | `/triage` | JSON: patient fields + optional `image_result` | `{risk, triage, center:{name, level, travel_km, travel_h, out_of_pocket_fcfa, …}}` |

Example:

```bash
curl -F image=@scan.png localhost:5000/classify
# {"class":"malignant","confidence":0.997,...}

curl -s localhost:5000/triage -H 'Content-Type: application/json' -d '{
  "age":62,"family_history":1,"palpability":1,"prior_biopsy":0,
  "symptom_duration_months":12,"rural_flag":1,"wealth_quartile":1,
  "region":"Bamako",
  "image_result":{"confidence":0.95,"above_floor":true}
}'
# {"risk":...,"triage":"urgent_biopsy","center":{...Bamako hospital...}}
```

---

## 9. Limitations & next steps

- Patient cohort is **synthetic** (2,000 records); imaging is real (BUSI). Clinical
  deployment needs a prospective Mali cohort — and re-running the F6 fairness audit on
  real subgroups first.
- int8 phone model is at 0.803 vs 0.940 fp32 — closing that gap needs QAT inside the
  TF/TFLite build (F3's recipe, once a healthy TF environment is available).
- The fairness region/wealth gaps (F6) are release blockers as-is.
- `keras .h5` cannot be reloaded in this environment (F2b) — use the SavedModel dir.

---

## 10. License

MIT — see [LICENSE](LICENSE).
