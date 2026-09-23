"""Grad-CAM on the torch MobileNetV3-Small imaging model.

Targets the last convolutional block of ``model.features`` and backprops the
top predicted class score into its feature maps. Produces a [0,1] heatmap and
an optional overlay on the original image. Runs on CPU.
"""
from __future__ import annotations

import numpy as np

HEATMAP_TARGET_INDEX = -1  # last block of MobileNetV3 features

CLASSES = ["benign", "malignant", "normal"]


def load_model(path: str = "models/imaging.pt", num_classes: int = 3, device: str = "cpu"):
    import torch

    from src.imaging.classifier import build_model

    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def gradcam(model, tensor, device: str = "cpu"):
    """Return (cam, class_index, confidence) — cam is [H,W] floats in [0,1]."""
    import torch
    from torch.nn.functional import softmax

    model = model.to(device)
    tensor = tensor.to(device)
    target = model.features[HEATMAP_TARGET_INDEX]

    activations: list = []
    gradients: list = []
    h_fwd = target.register_forward_hook(lambda _m, _i, o: activations.append(o))
    h_bwd = target.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))

    model.zero_grad()
    with torch.set_grad_enabled(True):
        out = model(tensor)
        idx = int(out[0].argmax(0).detach())
        out[0, idx].backward()
    conf = float(softmax(out[0], dim=0)[idx].detach())

    h_fwd.remove()
    h_bwd.remove()

    a = activations[0][0].detach().cpu()   # [C, H, W]
    g = gradients[0][0].detach().cpu()     # [C, H, W]
    weights = g.mean(dim=(1, 2))           # global-average-pooled gradients, [C]
    cam = torch.relu((weights[:, None, None] * a).sum(0)).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, idx, conf


def overlay(image, cam, alpha: float = 0.45, resize=(224, 224)):
    """Blend a heatmap over a PIL image; returns a new PIL image (no cv2 needed)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        rgb = Image.fromarray(image).convert("RGB")
    heat_resized = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize(rgb.size)
    heat_rgb = plt.get_cmap("jet")(np.asarray(heat_resized) / 255.0)[..., :3]  # [H,W,3] floats
    base = np.asarray(rgb).astype(np.float32) / 255.0
    blend = np.clip(base * (1 - alpha) + heat_rgb * alpha, 0, 1)
    return Image.fromarray((blend * 255).astype(np.uint8))


def preprocess(image_bytes: bytes):
    """Return the model-ready [1,3,224,224] tensor, same preprocessing as /classify."""
    import io

    import torch
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img, tf(img).unsqueeze(0)