"""Generate synthetic training data for the mama-sante bootstrap.

Writes data/raw/patients.csv and data/geo/centers.csv.
Label distribution is calibrated to be class-imbalanced (malignant ~8-10%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_PATIENTS = 2000

CITIES = [
    ("Bamako", 12.6392, -8.0029, "hospital"),
    ("Sikasso", 11.3160, -5.6660, "regional"),
    ("Segou", 13.4317, -6.2157, "regional"),
    ("Koulikoro", 12.8627, -7.5599, "district"),
    ("Kayes", 14.4465, -11.4415, "regional"),
    ("Mopti", 14.4874, -4.1835, "district"),
    ("San", 13.3034, -4.8999, "district"),
    ("Koutiala", 12.3917, -5.4644, "district"),
    ("Gao", 16.2704, -0.0447, "district"),
    ("Timbuktu", 16.7735, -3.0071, "post"),
    ("Nioro du Sahel", 15.2329, -9.5933, "post"),
    ("Kidal", 18.4411, 1.4077, "post"),
]

EQUIPMENT = {
    "post": ["us"],
    "district": ["us", "biopsy"],
    "regional": ["us", "mammo", "biopsy"],
    "hospital": ["us", "mammo", "biopsy", "surgery"],
}


def make_patients() -> pd.DataFrame:
    age = np.clip(RNG.normal(loc=48, scale=12, size=N_PATIENTS), 18, 85).round()
    rural = RNG.binomial(1, 0.55, N_PATIENTS)
    wealth = RNG.integers(1, 5, N_PATIENTS)
    family_history = RNG.binomial(1, 0.12, N_PATIENTS)
    prior_biopsy = RNG.binomial(1, 0.05, N_PATIENTS)
    palpability = RNG.binomial(1, 0.15, N_PATIENTS)
    symptom_duration = RNG.exponential(4.0, N_PATIENTS).round(1)

    logit = (
        -4.0
        + 0.04 * (age - 50)
        + 1.2 * family_history
        + 0.9 * palpability
        + 0.6 * prior_biopsy
        + 0.15 * symptom_duration
        + 0.3 * rural
        - 0.1 * (wealth - 2)
    )
    prob = 1 / (1 + np.exp(-logit))
    label = RNG.binomial(1, prob)

    patient_id = [f"ML-{i:06d}" for i in range(1, N_PATIENTS + 1)]
    return pd.DataFrame(
        {
            "patient_id": patient_id,
            "age": age.astype(int),
            "rural_flag": rural.astype(int),
            "wealth_quartile": wealth.astype(int),
            "family_history": family_history.astype(int),
            "prior_biopsy": prior_biopsy.astype(int),
            "palpability": palpability.astype(int),
            "symptom_duration_months": symptom_duration,
            "region": pd.Series(patient_id).map(lambda _: str(_)).str[:0],
            "label": label.astype(int),
        }
    ).assign(region=pd.Series(RNG.choice([c[0] for c in CITIES], N_PATIENTS)))


def make_centers() -> pd.DataFrame:
    base_cost = {"post": 5000, "district": 15000, "regional": 40000, "hospital": 80000}
    rows = []
    for i, (name, lat, lon, level) in enumerate(CITIES, start=1):
        rows.append(
            {
                "center_id": i,
                "name": name,
                "lat": lat,
                "lon": lon,
                "level": level,
                "equipment": ",".join(EQUIPMENT[level]),
                "capacity": int(RNG.integers(20, 200)),
                "base_cost": base_cost[level],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    patients = make_patients()
    centers = make_centers()

    patients.to_csv("data/raw/patients.csv", index=False)
    centers.to_csv("data/geo/centers.csv", index=False)

    malignant = patients["label"].mean()
    print(f"patients: {len(patients)} rows, {centers.shape[0]} centers")
    print(f"malignant rate: {malignant:.1%}")
    print(patients[["label", "rural_flag", "family_history"]].mean())