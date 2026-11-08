"""Pretrained ResNet-50 features (no training or fine-tuning).

The ImageNet classification layer is replaced by an identity, so the
network outputs the 2,048 values of its global average pooling layer.
"""

import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50

from .data import load_image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    return model.eval()


def to_tensor(img_uint8):
    """224x224 RGB uint8 -> normalised (3, 224, 224) tensor."""
    x = (img_uint8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


@torch.no_grad()
def extract(paths, batch_size=64, log_every=10):
    model = build_model()
    out = []
    for b, start in enumerate(range(0, len(paths), batch_size)):
        batch = torch.stack([to_tensor(load_image(p)) for p in paths[start:start + batch_size]])
        out.append(model(batch).numpy())
        if b % log_every == 0:
            print(f"  {start + len(batch)}/{len(paths)} images")
    return np.concatenate(out).astype(np.float32)
