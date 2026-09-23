import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.imaging import gradcam  # noqa: E402

CLASSES = gradcam.CLASSES


@pytest.fixture(scope="module")
def model():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "imaging.pt")
    return gradcam.load_model(path)


def test_gradcam_shapes(model, real_image):
    pil, tensor = gradcam.preprocess(real_image)
    cam, idx, conf = gradcam.gradcam(model, tensor)
    assert CLASSES[idx] in CLASSES
    assert 0.0 <= conf <= 1.0
    assert cam.ndim == 2
    assert cam.shape == (7, 7)  # MobileNetV3-Small features[-1] output
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-8


def test_gradcam_decision_boundary_not_all_zero(model, real_image):
    _, tensor = gradcam.preprocess(real_image)
    cam, _, _ = gradcam.gradcam(model, tensor)
    assert np.unique(cam).size > 1  # a real heatmap, not uniform


def test_overlay_returns_same_size(model, real_image):
    pil, tensor = gradcam.preprocess(real_image)
    cam, _, _ = gradcam.gradcam(model, tensor)
    out = gradcam.overlay(pil, cam)
    assert out.size == pil.size
    assert isinstance(out, Image.Image)


def test_fallback_for_non_png_payload():
    # preprocess must still yield a 1x3x224x224 tensor for a JPEG-style buffer made via PIL
    import io

    buf = io.BytesIO()
    Image.new("RGB", (600, 400), (80, 80, 80)).save(buf, "JPEG")
    _, tensor = gradcam.preprocess(buf.getvalue())
    assert tuple(tensor.shape) == (1, 3, 224, 224)