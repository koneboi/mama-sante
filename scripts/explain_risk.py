"""CLI: local explainability for one patient's risk score.

Example:
    python scripts/explain_risk.py --age 62 --family-history 1 --palpability 1 \
        --symptom-duration-months 12 --region Koulikoro --wealth-quartile 1 --rural 1
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.risk.explain import explain  # noqa: E402
from src.risk.stratifier import FEATURES, load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for f in FEATURES:
        ap.add_argument("--" + f.replace("_", "-"), type=float, default=0.0)
    ap.add_argument("--version", action="version", version="%(prog)s 1.0")
    args = ap.parse_args()

    row = pd.DataFrame([{f: getattr(args, f.replace("-", "_")) for f in FEATURES}])
    model = load(os.path.join(ROOT, "models", "risk_gb.pkl"))
    cohort = pd.read_csv(os.path.join(ROOT, "data", "raw", "patients.csv"))

    out = explain(model, row, FEATURES, cohort=cohort)
    print(f"method       : {out['method']}")
    print(f"baseline risk: {out['baseline_risk']:.4f}")
    print(f"risk score   : {out['risk_score']:.4f}")
    print("factors (by |contribution|):")
    for f in out["factors"]:
        print(f"  {f['feature']:<26} value {f['value']:>8.3f}  baseline {f['baseline']:>8.3f}  contrib {f['contribution']:+.5f}")


if __name__ == "__main__":
    main()