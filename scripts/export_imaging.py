"""Export the trained imaging model to ONNX for low-end / on-device inference."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(checkpoint: str = "models/imaging.pt", out_path: str = "models/imaging.onnx") -> None:
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from torchvision import transforms

    from src.imaging.classifier import build_model

    state = torch.load(checkpoint, map_location="cpu")
    model = build_model(num_classes=state["num_classes"], pretrained=False)
    model.load_state_dict(state["model"])
    model.eval()

    wrapper = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["probs"],
        opset_version=13,
        dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}},
    )
    print(f"exported -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    # parity + latency check on ONNX runtime. NOTE: torch.onnx.export mutates the
    # traced model in-memory (torch 2.8), so load a fresh copy for comparison.
    fresh = build_model(num_classes=state["num_classes"], pretrained=False)
    fresh.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    fresh.eval()
    fresh_wrapper = torch.nn.Sequential(fresh, torch.nn.Softmax(dim=1))

    # parity + latency check on ONNX runtime
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    sample = [p for p in __import__("glob").glob(f"{ROOT}/data/processed/busi_class/*/*.png")][:8]
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    import time

    max_diff, t0 = 0.0, time.time()
    for p in sample:
        img = tf(Image.open(p).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        with torch.no_grad():
            torch_probs = fresh_wrapper(torch.from_numpy(img)).numpy()
        ort_probs = sess.run(None, {"image": img})[0]
        max_diff = max(max_diff, float(np.abs(torch_probs - ort_probs).max()))
    ms = (time.time() - t0) / len(sample) * 1000
    print(f"max |torch - onnx| diff over {len(sample)} samples: {max_diff:.2e}")
    print(f"avg CPU latency: {ms:.1f} ms/image (onnxruntime fp32)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/imaging.pt")
    p.add_argument("--output", default="models/imaging.onnx")
    args = p.parse_args()
    main(args.checkpoint, args.output)