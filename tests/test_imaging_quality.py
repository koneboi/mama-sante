import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.imaging.quality import assess_image  # noqa: E402


def test_real_ultrasound_passes(real_image):
    r = assess_image(real_image)
    assert r["ok"] is True
    assert r["reasons"] == []
    assert r["checks"]["luminance_std"] > 0
    assert r["checks"]["colorfulness"] < 0.10


def test_solid_white_rejected(white_image):
    r = assess_image(white_image)
    assert not r["ok"]
    assert "flat_contrast" in r["reasons"]


def test_rgb_noise_rejected(noise_image):
    r = assess_image(noise_image)
    assert not r["ok"]
    assert "not_grayscale" in r["reasons"]


def test_tiny_image_rejected(tiny_image):
    r = assess_image(tiny_image)
    assert not r["ok"]
    assert "too_small" in r["reasons"]


def test_garbage_bytes_rejected():
    r = assess_image(b"not an image at all")
    assert not r["ok"]
    assert r["reasons"] == ["undecodable"]


def test_missing_guard_off_path_unused(real_image):
    # OOD_GUARD is read from config; quality itself does not depend on it
    r = assess_image(real_image)
    assert r["ok"]