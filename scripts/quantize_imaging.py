"""Export a static int8-quantized ONNX model and verify it against the fp32 model.

Calibration uses a few hundred training images from data/processed/busi_class;
evaluation uses the same 15% holdout split the trainer used.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class CalibrationReader:
    def __init__(self, paths: list[str], batch_size: int = 8):
        self._it = self._gen(paths, batch_size)

    def _gen(self, paths, batch_size):
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            imgs = np.stack([TF(Image.open(p).convert("RGB")).numpy() for p in batch])
            yield {"image": imgs.astype(np.float32)}

    def get_next(self):
        return next(self._it, None)


def split_paths(root: str, val_ratio: float = 0.15, seed: int = 0):
    rng = random.Random(seed)
    paths = sorted(glob.glob(os.path.join(root, "*", "*.png")))
    rng.shuffle(paths)
    n = int(len(paths) * (1 - val_ratio))
    return paths[:n], paths[n:]


def main(onnx_path: str = "models/imaging.onnx", out_path: str = "models/imaging_int8.onnx") -> None:
    from onnxruntime.quantization import QuantType, quantize_static

    train_paths, val_paths = split_paths("data/processed/busi_class")
    assert train_paths and val_paths
    print(f"calibration: {len(train_paths)} train / {len(val_paths)} val holdout")

    quantize_static(
        onnx_path,
        out_path,
        calibration_data_reader=CalibrationReader(train_paths[:200]),
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"saved -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB vs "
          f"{os.path.getsize(onnx_path) / 1e6:.1f} MB fp32)")

    sess_fp32 = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    sess_i8 = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])

    indices = [0] * 3
    y_true = []
    correct = {"fp32": 0, "i8": 0}
    agree = 0
    for i, p in enumerate(val_paths):
        img = TF(Image.open(p).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        probs_f = sess_fp32.run(None, {"image": img})[0]
        probs_i = sess_i8.run(None, {"image": img})[0]
        cls_folder = os.path.basename(os.path.dirname(p))
        class_idx = {"benign": 0, "malignant": 1, "normal": 2}[cls_folder]
        y_true.append(class_idx)
        correct["fp32"] += probs_f.argmax(1)[0] == class_idx
        correct["i8"] += probs_i.argmax(1)[0] == class_idx
        agree += probs_f.argmax(1)[0] == probs_i.argmax(1)[0]

    n = len(y_true)
    fp32_acc, i8_acc = correct["fp32"] / n, correct["i8"] / n
    print(f"val acc  fp32: {fp32_acc:.3f}   int8: {i8_acc:.3f}")
    print(f"argmax agreement fp32<->int8: {agree / n:.3f}")

    images = val_paths[:20]
    times = {}
    for name, sess in [("fp32", sess_fp32), ("int8", sess_i8)]:
        imgs = [TF(Image.open(p).convert("RGB")).unsqueeze(0).numpy().astype(np.float32) for p in images]
        t0 = time.time()
        runs = 10
        for _ in range(runs):
            for img in imgs:
                sess.run(None, {"image": img})
        times[name] = (time.time() - t0) / (len(imgs) * runs) * 1000
    print(f"CPU latency  fp32: {times['fp32']:.1f} ms   int8: {times['int8']:.1f} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="models/imaging.onnx")
    p.add_argument("--output", default="models/imaging_int8.onnx")
    a = p.parse_args()
    main(a.onnx, a.output)