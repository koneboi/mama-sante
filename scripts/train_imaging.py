"""Transfer-learn a MobileNet classifier on breast ultrasound images.

Expects --data as a folder path with one subdir per class, named after the class
index or a 0/1 prefix, e.g.:

    --data data/processed/busi
        benign/*.png
        malignant/*.png

Run a self-contained smoke test (generates synthetic images, no download):
    python3 scripts/train_imaging.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image, ImageDraw  # noqa: E402


def _make_synthetic_image(path: str, label: int, seed: int) -> None:
    rng = random.Random(seed)
    img = Image.new("L", (224, 224), 0)
    d = ImageDraw.Draw(img)
    cx, cy, r = rng.randint(80, 144), rng.randint(80, 144), rng.randint(45, 70)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=120)
    for _ in range(rng.randint(6, 14)):
        s = rng.randint(2, 12)
        x0, y0 = rng.randint(0, 224 - s), rng.randint(0, 224 - s)
        d.ellipse([x0, y0, x0 + s, y0 + s], fill=rng.randint(60, 130))
    if label == 1:
        rl = rng.randint(8, 16)
        rx, ry = rng.randint(90, 134), rng.randint(90, 134)
        d.ellipse([rx - rl, ry - rl, rx + rl, ry + rl], fill=255)
    img.save(path)


def _make_smoke_data(root: str, per_class: int = 40) -> list[str]:
    classes = {0: "benign", 1: "malignant"}
    for label, name in classes.items():
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        for i in range(per_class):
            _make_synthetic_image(os.path.join(d, f"{i}.png"), label, seed=label * 1000 + i)
    return [os.path.join(root, c) for c in classes.values()]


def train(data_dir: str, num_classes: int, epochs: int, batch_size: int, out_path: str, device: str):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, transforms

    from src.imaging.classifier import build_model

    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ds = datasets.ImageFolder(data_dir, transform=train_tf)
    # deterministic per-class 85/15 train/val split
    rng = random.Random(0)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    n_train = int(len(idxs) * 0.85)
    train_idx, val_idx = set(idxs[:n_train]), idxs[n_train:]
    train_ds = torch.utils.data.Subset(ds, list(train_idx))
    val_ds = torch.utils.data.Subset(
        datasets.ImageFolder(data_dir, transform=val_tf), val_idx
    )

    targets = torch.tensor(ds.targets)[list(train_idx)]
    counts = torch.bincount(targets).float()
    weights = counts.reciprocal()[targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))
    loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=num_classes, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    n = len(train_ds)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            running_loss += loss.item() * y.size(0)

        model.eval()
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_correct += (model(x).argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        printed = (
            f"epoch {epoch:02d}/{epochs} "
            f"loss {running_loss / n:.4f} train_acc {(correct / total):.3f} "
            f"val_acc {val_acc:.3f}"
        )
        print(printed, flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": num_classes,
                    "classes": ds.classes,
                    "val_acc": val_acc,
                },
                out_path,
            )

    meta_path = out_path.replace(".pt", ".json")
    with open(meta_path, "w") as f:
        json.dump({"num_classes": num_classes, "data_dir": data_dir, "epochs": epochs, "best_val_acc": best_acc}, f)
    print(f"best val_acc {best_acc:.3f} -> {out_path}")
    print(f"classes: {ds.classes}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/busi")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="models/imaging.pt")
    p.add_argument("--smoke", action="store_true", help="train on synthetic images")
    args = p.parse_args()

    if args.smoke:
        data_dir = os.path.join(ROOT, "data", "processed", "_smoke")
        _make_smoke_data(data_dir)
        train(data_dir, 2, epochs=2, batch_size=8, out_path=args.output, device="cpu")
    else:
        if not os.path.isdir(args.data):
            sys.exit(f"data dir not found: {args.data}\nDownload BUSI and point --data at it.")
        t0 = time.time()
        train(args.data, args.num_classes, args.epochs, args.batch_size, args.output, "cuda" if __import__("torch").cuda.is_available() else "cpu")
        print(f"training took {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()