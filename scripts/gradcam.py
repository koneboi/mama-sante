"""CLI: produce a Grad-CAM overlay for one ultrasound image.

Example:
    python scripts/gradcam.py --image data/processed/busi_class/benign/benign\ \(1\).png --out outputs/gradcam_benign.png
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "gradcam.png"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "imaging.pt"))
    ap.add_argument("--save-heatmap", action="store_true", help="also save the raw heatmap as a .npy")
    args = ap.parse_args()

    from src.imaging.gradcam import CLASSES, gradcam, load_model, overlay, preprocess

    with open(args.image, "rb") as f:
        img_bytes = f.read()
    pil, tensor = preprocess(img_bytes)
    model = load_model(args.model)
    cam, pred_idx, conf = gradcam(model, tensor)

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    overlay(pil, cam).save(args.out)
    print(f"prediction: {CLASSES[pred_idx]} (confidence {conf:.3f})")
    print(f"heatmap shape {cam.shape} range [{cam.min():.2f}, {cam.max():.2f}]")
    print(f"saved overlay -> {args.out}")
    if args.save_heatmap:
        npy = args.out.rsplit(".", 1)[0] + ".heatmap.npy"
        np.save(npy, cam)
        print(f"saved heatmap -> {npy}")


if __name__ == "__main__":
    main()