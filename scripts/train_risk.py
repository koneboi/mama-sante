"""Fit the risk stratifier on the patient records and persist it."""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.risk.stratifier import FEATURES, TARGET, load, save, train


def main(data_path: str = "data/raw/patients.csv", out_path: str = "models/risk_gb.pkl") -> None:
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]

    t0 = time.time()
    model = train(df)
    save(model, out_path)
    fit_h = (time.time() - t0) / 60
    print(f"fitted + saved -> {out_path} ({fit_h:.2f} min)")

    scores = cross_val_score(
        model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc"
    )
    print(f"5-fold CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    sanity = load(out_path)
    print(f"reload sanity: {type(sanity).__name__}")


if __name__ == "__main__":
    main()