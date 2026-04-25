"""
agent/ml/deepfake_efficientnet.py — EfficientNet-B0 Deepfake Detector
======================================================================
Model:   EfficientNet-B0  (torchvision, ImageNet pretrained backbone)
Weights: models/deepfake_efficientnet_final.pt
         Trained locally on:
           - Celeb-DF v2 (real + fake)
           - Webcam real captures (person_1..4)
           - OBS virtual camera fakes (obs_p1..4)
         Domain-aligned, identity-safe train/val split.

Validation results (held-out val set):
  REAL avg P(fake) = 0.127   (FP rate = 6.7%)
  FAKE avg P(fake) = 0.788   (detection rate = 72.5%)
  Val accuracy     = 93.7%

Architecture:
  efficientnet_b0
    classifier[1]: Linear(1280 -> 1)   single logit

Output:
  sigmoid(logit).item()   -> P(fake), float in [0, 1]

Preprocessing (EXACT match to training — agent/scripts/train_deepfake_final.py):
  BGR numpy
    -> RGB
    -> PIL.Image.fromarray()
    -> Resize(224, 224)
    -> ToTensor()                [0,255] -> [0.0, 1.0]
    -> Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])  (ImageNet)

IMPORTANT: Normalize IS used (ImageNet stats) — model trained with it.

This module is PURE inference — no async, no events, no throttle.
All of that lives in DeepfakeInferenceEngine (deepfake_inference.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Local weights path ─────────────────────────────────────────────────────────
WEIGHTS_LOCAL = Path("models/deepfake_efficientnet_final.pt")

# ImageNet normalization stats — MUST match training
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model(device: str = "cpu", weights_path: Path = WEIGHTS_LOCAL):
    """
    Build EfficientNet-B0 with binary head (Linear 1280->1) and load
    locally-trained weights from models/deepfake_efficientnet_final.pt.

    Returns:
      model in eval() mode on `device`, or None on failure.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import efficientnet_b0

        # ── Build architecture (MUST match training: Linear(1280->1)) ────────
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features          # 1280
        model.classifier[1] = nn.Linear(in_features, 1)        # single logit

        # ── Load local weights ────────────────────────────────────────────────
        if not weights_path.exists():
            print(
                f"EFFICIENTNET: weights not found at {weights_path}\n"
                f"  Run: python scripts/train_deepfake_final.py to train."
            )
            return None

        print(f"EFFICIENTNET: loading from {weights_path}")
        state = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(state)
        model.eval()
        model.to(device)

        size_mb = weights_path.stat().st_size / 1e6
        print(
            f"EFFICIENTNET: model ready  device={device}  "
            f"size={size_mb:.1f}MB  head=Linear({in_features}->1)"
        )
        return model

    except Exception as exc:
        print(f"EFFICIENTNET: load_model FAILED — {exc}")
        logger.error(f"EfficientNet load failed: {exc}", exc_info=True)
        return None


def preprocess(bgr_numpy: np.ndarray):
    """
    Convert BGR face crop to EfficientNet input tensor.

    Preprocessing EXACTLY as used during training
    (scripts/train_deepfake_final.py, _base transform):

      BGR ndarray
        -> RGB  (channel flip)
        -> PIL.Image.fromarray()
        -> Resize(224, 224)
        -> ToTensor()          [0,255] uint8 -> [0.0, 1.0] float
        -> Normalize(ImageNet mean, std)
        -> unsqueeze(0)        add batch dim

    Returns:
      torch.Tensor  shape [1, 3, 224, 224], float32
    """
    from PIL import Image
    from torchvision import transforms

    rgb = bgr_numpy[..., ::-1].copy()           # BGR -> RGB
    pil = Image.fromarray(rgb.astype("uint8"))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return tf(pil).unsqueeze(0)                  # [1, 3, 224, 224]


def predict(bgr_numpy: np.ndarray, model, device: str = "cpu") -> float:
    """
    Run EfficientNet-B0 deepfake detection on a single face crop.

    Args:
      bgr_numpy: OpenCV BGR face crop, uint8, any resolution.
      model:     loaded model from load_model().
      device:    'cpu' or 'cuda'.

    Returns:
      float: P(fake) in [0, 1].
        0.0 = confidently real
        1.0 = confidently fake
    """
    import torch

    tensor = preprocess(bgr_numpy).to(device)
    with torch.no_grad():
        logit = model(tensor)                        # [1, 1]
    prob_fake = torch.sigmoid(logit)[0][0].item()   # single logit -> sigmoid
    return float(prob_fake)
