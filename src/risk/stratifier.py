import joblib
import pandas as pd

FEATURES = [
    "age",
    "family_history",
    "prior_biopsy",
    "palpability",
    "symptom_duration_months",
    "rural_flag",
    "wealth_quartile",
]
TARGET = "label"


def load_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train(df: pd.DataFrame, feature_cols: list[str] = None, target_col: str = TARGET):
    """Train the tabular risk stratifier. Returns a fitted sklearn model."""
    from sklearn.ensemble import GradientBoostingClassifier

    feature_cols = feature_cols or FEATURES
    model = GradientBoostingClassifier(random_state=0)
    model.fit(df[feature_cols], df[target_col])
    return model


def score(model, df: pd.DataFrame, feature_cols: list[str] = None) -> pd.Series:
    """Probability of high risk (class 1)."""
    feature_cols = feature_cols or FEATURES
    return pd.Series(
        model.predict_proba(df[feature_cols])[:, 1],
        index=df.index,
        name="risk_score",
    )


def save(model, path: str = "models/risk_gb.pkl") -> str:
    joblib.dump(model, path)
    return path


def load(path: str = "models/risk_gb.pkl"):
    return joblib.load(path)