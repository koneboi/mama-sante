def build_model(num_classes: int = 3, pretrained: bool = True):
    """MobileNetV3-Small classifier head. Kept light for offline devices."""
    import torch.nn as nn
    from torchvision import models

    model = models.mobilenet_v3_small(pretrained=pretrained)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def predict(model, image_tensor, confidence_floor: float) -> dict:
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(image_tensor), dim=-1)
    conf, pred = probs.max(dim=-1)
    return {
        "pred_class": int(pred),
        "confidence": float(conf),
        "above_floor": bool(conf >= confidence_floor),
    }
