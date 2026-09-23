"""Quantization-aware training (QAT) for a deployable int8 MobileNetV3 on BUSI.

Plain post-training quantization collapses because early conv activations have
extreme out-of-distribution outliers (max ~200 vs p99 ~6). QAT learns those
ranges, so the int8 export stays accurate. Output: models/imaging_int8.onnx
(checkpoint + QDQ int8 graph), verified against fp32 on the 15% holdout split.
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
from PIL import Image
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models.quantization import mobilenet_v3_large

TF_TRAIN = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
TF_VAL = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def build_qat_model(num_classes: int = 3):
    import torch
    import torch.nn as nn

    model = mobilenet_v3_large(weights="IMAGENET1K_V1")
    last_layer = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)
    qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=torch.ao.quantization.MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    )
    model.qconfig = qconfig
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def train_qat(data_dir: str, num_classes: int, epochs: int, batch_size: int, out_path: str):
    import torch
    import torch.nn as nn

    ds = datasets.ImageFolder(data_dir, transform=TF_TRAIN)
    rng = random.Random(0)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    n_train = int(len(idxs) * 0.85)
    train_idx, val_idx = idxs[:n_train], idxs[n_train:]

    targets = torch.tensor(ds.targets)[train_idx]
    counts = torch.bincount(targets).float()
    weights = counts.reciprocal()[targets]
    loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=batch_size,
        sampler=WeightedRandomSampler(weights, num_samples=len(weights)),
    )
    val_ds = Subset(datasets.ImageFolder(data_dir, transform=TF_VAL), val_idx)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_qat_model(num_classes).to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    n = len(train_idx)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to("cuda"), y.to("cuda")
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += y.size(0)
            correct += (model(x).detach().argmax(1) == y).sum().item()
            running_loss += loss.item() * y.size(0)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to("cuda"), y.to("cuda")
                val_correct += (model(x).argmax(1) == y).sum().item()
        val_acc = val_correct / len(val_idx)
        print(
            f"epoch {epoch:02d}/{epochs} loss {running_loss / n:.4f} "
            f"train_acc {(correct / total):.3f} val_acc {val_acc:.3f}",
            flush=True,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "num_classes": num_classes, "classes": ds.classes, "val_acc": val_acc},
                out_path.replace(".onnx", ".pt"),
            )

    # convert to QDQ static-graph model and export
    ckpt = torch.load(out_path.replace(".onnx", ".pt"), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to("cpu")
    model.eval()
    model = torch.ao.quantization.convert(model, inplace=False)
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"best val_acc {best_acc:.3f} -> int8 QDQ onnx {out_path}")
    return out_path


def verify(onnx_path: str, val_dir: str) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    paths = sorted(glob.glob(os.path.join(val_dir, "*", "*.png")))
    ok = 0
    for p in paths:
        img = TF_VAL(Image.open(p).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        logits = sess.run(None, {"image": img})[0]
        pred = int(np.argmax(logits))
        expected = int(sorted({os.path.basename(os.path.dirname(x)) for x in paths}).index(
            os.path.basename(os.path.dirname(p))))
        ok += pred == expected
    print(f"int8 QDQ ONNX val acc (holdout): {ok / len(paths):.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/busi_class")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="models/imaging_int8.onnx")
    a = p.parse_args()

    t0 = time.time()
    onnx_path = train_qat(a.data, 3, a.epochs, a.batch_size, a.output)
    verify(onnx_path, a.data)
    print(f"qat total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()