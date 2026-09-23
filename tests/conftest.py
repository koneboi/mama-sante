import glob
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def busi_image_path() -> str:
    files = sorted(glob.glob(os.path.join(ROOT, "data", "processed", "busi_class", "*", "*.png")))
    assert files, "no BUSI image found under data/processed/busi_class"
    return files[0]


@pytest.fixture()
def real_image(busi_image_path) -> bytes:
    with open(busi_image_path, "rb") as f:
        return f.read()


@pytest.fixture()
def white_image() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (500, 500), (255, 255, 255)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def noise_image() -> bytes:
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (500, 500, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def tiny_image() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), (128, 128, 128)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def cohort() -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "raw", "patients.csv")
    assert os.path.exists(path), f"missing {path}"
    return pd.read_csv(path)


@pytest.fixture()
def app_client():
    from src.api.app import app

    app.config.update(TESTING=True)
    return app.test_client()