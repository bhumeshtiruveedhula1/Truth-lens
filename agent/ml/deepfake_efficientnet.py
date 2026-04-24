"""
agent/ml/deepfake_efficientnet.py — EfficientNet-B0 Deepfake Detector
======================================================================
Model:   EfficientNet-B0  (torchvision)
Weights: Xicor9/efficientnet-b0-ffpp-c23  (HuggingFace, FF++ C23)
         ~21 MB, downloaded via torch.hub on first use, cached to disk.

Training data (from model card):
  FaceForensics++ C23 — DeepFake, FaceSwap, Face2Face, NeuralTextures

Architecture (exact match required for state_dict load):
  efficientnet_b0
    └── classifier[1]: Linear(1280 → 2)   [0=Real, 1=Fake]

Output:
  softmax(logits, dim=1)[0][1].item()   → P(fake), float in [0, 1]

Preprocessing (EXACT from model card — NO normalization):
  BGR numpy → RGB → PIL
  → Resize(224, 224)
  → ToTensor()          [0,255] → [0.0, 1.0]

This module is PURE inference — no async, no events, no throttle.
All of that lives in DeepfakeInferenceEngine (deepfake_inference.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Remote weights ─────────────────────────────────────────────────────────────
WEIGHTS_URL = (
    "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23"
    "/resolve/main/efficientnet_b0_ffpp_c23.pth"
)

# Local cache path — avoids re-downloading on every restart
WEIGHTS_CACHE = Path("models/deepfake_efficientnet_b0.pt")


def load_model(device: str = "cpu", cache_path: Path = WEIGHTS_CACHE):
    """
    Build EfficientNet-B0 with binary deepfake head and load FF++ weights.

    Download order:
      1. cache_path (local disk)  — fast, offline-safe
      2. WEIGHTS_URL (HuggingFace) — first-run download (~21 MB)

    Returns:
      model in eval() mode on `device`, or None on failure.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import efficientnet_b0

        # ── Build architecture (must match checkpoint exactly) ────────────────
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features          # 1280
        model.classifier[1] = nn.Linear(in_features, 2)        # [Real, Fake]

        # ── Load weights ──────────────────────────────────────────────────────
        if cache_path.exists():
            print(f"EFFICIENTNET: loading from cache  {cache_path}")
            state = torch.load(str(cache_path), map_location=device)
        else:
            print(
                f"EFFICIENTNET: downloading FF++ weights (~21 MB)...\n"
                f"  from: {WEIGHTS_URL}\n"
                f"  cache: {cache_path}"
            )
            state = torch.hub.load_state_dict_from_url(
                WEIGHTS_URL,
                map_location=device,
                progress=True,
            )
            # Save to cache for future runs
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, str(cache_path))
            print(f"EFFICIENTNET: weights cached to {cache_path}")

        model.load_state_dict(state)
        model.eval()
        model.to(device)
        print(f"EFFICIENTNET: model ready  device={device}  in_features={in_features}")
        return model

    except Exception as exc:
        print(f"EFFICIENTNET: load_model FAILED — {exc}")
        logger.error(f"EfficientNet load failed: {exc}", exc_info=True)
        return None


def preprocess(bgr_numpy: np.ndarray):
    """
    Convert BGR face crop to EfficientNet input tensor.

    Preprocessing EXACTLY as defined in model card
    (https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23):

      BGR ndarray
        → RGB  (channel flip)
        → PIL.Image.fromarray()
        → Resize(224, 224)
        → ToTensor()          [0,255] uint8 → [0.0, 1.0] float
        → unsqueeze(0)        add batch dim

    IMPORTANT: NO ImageNet normalization — the model was trained without it.

    Returns:
      torch.Tensor  shape [1, 3, 224, 224], float32, CPU
    """
    from PIL import Image
    from torchvision import transforms

    rgb = bgr_numpy[..., ::-1].copy()           # BGR → RGB
    pil = Image.fromarray(rgb.astype("uint8"))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),                   # [0,1] only — no Normalize()
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
    import torch.nn as nn

    tensor = preprocess(bgr_numpy).to(device)
    with torch.no_grad():
        logits = model(tensor)                       # [1, 2]
    prob_fake = torch.softmax(logits, dim=1)[0][1].item()
    return float(prob_fake)
