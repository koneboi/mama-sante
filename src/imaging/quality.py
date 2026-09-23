"""Cheap structural image-quality gate for the ultrasound /classify endpoint.

The gate runs before inference and rejects inputs that an ultrasound property
cannot plausibly produce: undecodable bytes, tiny images, flat/low-contrast
frames (blank or gradient), strongly coloured natural photos, and fully
clipped (solid black/white) frames. Ultrasound images are grayscale, so high
inter-channel colour spread is a strong non-ultrasound signal; this is exactly
the family of OOD inputs whose conv activations collapsed ORT int8 (max 198 vs
p99 6). Rejects are reported to the caller without running the model, so
garbage can never come back with a confident class.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

MIN_SIDE = 64
_MIN_CONTRAST = 0.04        # luminance std on 0-1 scale
_MAX_COLORFULNESS = 0.10    # mean abs inter-channel spread on 0-1 scale
_MAX_CLIPPED_FRACTION = 0.98


def decode(image_bytes: bytes) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def assess_image(image_bytes: bytes, min_side: int = MIN_SIDE) -> dict:
    """Return {"ok", "reasons", "checks"} — pure structural, no model needed."""
    img = decode(image_bytes)
    if img is None:
        return {"ok": False, "reasons": ["undecodable"], "checks": {}}

    w, h = img.size
    checks = {"width": w, "height": h}
    reasons: list[str] = []

    if min(w, h) < min_side:
        reasons.append("too_small")

    small = img.resize((128, 128))
    a = np.asarray(small, dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    checks["luminance_std"] = round(float(lum.std()), 4)
    if checks["luminance_std"] < _MIN_CONTRAST:
        reasons.append("flat_contrast")

    # colourfulness measured at full resolution: downsampling averages channel
    # noise together (white noise ~ 0.33 here collapses to ~0.07 at 128px) and
    # would hide non-ultrasound inputs behind the gate.
    full = np.asarray(img, dtype=np.float32) / 255.0
    checks["colorfulness"] = round(
        float(np.mean([np.abs(full[..., 0] - full[..., 1]).mean(),
                       np.abs(full[..., 1] - full[..., 2]).mean(),
                       np.abs(full[..., 0] - full[..., 2]).mean()])),
        4,
    )
    if checks["colorfulness"] > _MAX_COLORFULNESS:
        reasons.append("not_grayscale")

    clipped = float(((lum <= 1.0 / 255.0) | (lum >= 254.0 / 255.0)).mean())
    checks["clipped_fraction"] = round(clipped, 4)
    if clipped > _MAX_CLIPPED_FRACTION:
        reasons.append("clipped_extremes")

    return {"ok": not reasons, "reasons": reasons, "checks": checks}