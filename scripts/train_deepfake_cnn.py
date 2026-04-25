"""
scripts/train_deepfake_cnn.py
==============================
Training script for the domain-robust EfficientNet-B0 deepfake detector.

Dataset:  data/cnn_dataset/real/   (webcam real faces)
          data/cnn_dataset/fake/   (OBS + replay + deepfake faces)

Key design decisions (lessons from domain mismatch failures):
  1. Webcam-native dataset — model trained on same domain as runtime input
  2. Heavy compression + blur augmentation — simulates webcam degradation
  3. ImageNet normalization — required for pretrained EfficientNet backbone
  4. Temperature-based calibration after training — prevents overconfidence
  5. 80/20 train/val split stratified by label

Output:
  models/deepfake_efficientnet_b0_v2.pt   — fine-tuned weights
  models/deepfake_efficientnet_b0_v2_calibrated_T.txt — temperature value

Usage:
  python scripts/train_deepfake_cnn.py
  python scripts/train_deepfake_cnn.py --epochs 20 --batch 32 --lr 1e-4
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_deepfake")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data" / "cnn_dataset"
REAL_DIR      = DATA_DIR / "real"
FAKE_DIR      = DATA_DIR / "fake"
MODELS_DIR    = PROJECT_ROOT / "models"
OUTPUT_MODEL  = MODELS_DIR / "deepfake_efficientnet_b0_v2.pt"
OUTPUT_TEMP   = MODELS_DIR / "deepfake_efficientnet_b0_v2_calibrated_T.txt"

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULT_EPOCHS    = 15
DEFAULT_BATCH     = 32
DEFAULT_LR        = 1e-4
DEFAULT_VAL_SPLIT = 0.20
IMAGE_SIZE        = 224

# ── ImageNet normalization (required for pretrained backbone) ─────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Binary face-crop dataset.
      label 0 = REAL
      label 1 = FAKE

    Augmentation strategy (train only):
      - Random horizontal flip
      - Brightness + contrast jitter (webcam lighting variation)
      - Gaussian blur   (webcam focus blur + compression artifact)
      - JPEG recompression at quality 40-95 (critical for webcam domain match)
      - Gaussian noise injection
      - Random grayscale (5% — prevents color-based shortcuts)
    """

    # JPEG quality range for augmentation
    JPEG_QUALITY_RANGE = (40, 95)

    def __init__(
        self,
        real_paths: list[Path],
        fake_paths: list[Path],
        augment:    bool = True,
    ) -> None:
        self.samples: list[Tuple[Path, int]] = (
            [(p, 0) for p in real_paths] +
            [(p, 1) for p in fake_paths]
        )
        random.shuffle(self.samples)
        self.augment = augment

        # Base transform (always applied)
        self._base_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # Augmentation transforms (train only)
        self._aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Corrupted image — return black frame, don't crash training
            img = Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))

        if self.augment:
            img = self._aug_tf(img)
            img = self._jpeg_compress(img)
            img = self._add_noise(img)

        tensor = self._base_tf(img)
        return tensor, label

    def _jpeg_compress(self, img: Image.Image) -> Image.Image:
        """
        Re-encode image as JPEG at random quality then decode back.
        Simulates webcam stream compression artifacts.
        """
        quality = random.randint(*self.JPEG_QUALITY_RANGE)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()   # .copy() detaches from BytesIO

    @staticmethod
    def _add_noise(img: Image.Image, sigma_range=(0.0, 8.0)) -> Image.Image:
        """
        Add Gaussian noise to a PIL image.
        Simulates sensor noise in low-light or compressed webcam streams.
        """
        arr   = np.asarray(img, dtype=np.float32)
        sigma = random.uniform(*sigma_range)
        noise = np.random.randn(*arr.shape).astype(np.float32) * sigma
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(device: str, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B0 with binary deepfake head.

    Architecture:
      efficientnet_b0 (ImageNet pretrained)
        └── classifier[1]: Linear(1280 → 2)   [0=Real, 1=Fake]

    Fine-tuning strategy:
      - All layers unfrozen (full fine-tune)
      - Lower LR on backbone handles catastrophic forgetting
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model   = efficientnet_b0(weights=weights)

    # Replace classification head
    in_features            = model.classifier[1].in_features   # 1280
    model.classifier[1]    = nn.Linear(in_features, 2)

    model = model.to(device)
    logger.info(f"Model: EfficientNet-B0  head=Linear({in_features}→2)  device={device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Calibration — Temperature Scaling
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Single-parameter temperature scaling for post-hoc calibration.
    Optimized on validation set AFTER training to reduce overconfidence.

    Reference: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.5, max=10.0)


def calibrate_temperature(
    model:   nn.Module,
    loader:  DataLoader,
    device:  str,
) -> float:
    """
    Find optimal temperature T on the validation set using NLL loss.

    Returns:
      T (float) — optimal temperature value
    """
    model.eval()
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS(
        scaler.parameters(), lr=0.01, max_iter=50
    )
    criterion = nn.CrossEntropyLoss()

    # Collect all logits + labels from validation set (one pass)
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    def _closure():
        optimizer.zero_grad()
        scaled = scaler(all_logits)
        loss   = criterion(scaled, all_labels)
        loss.backward()
        return loss

    optimizer.step(_closure)

    T = float(scaler.temperature.item())
    logger.info(f"[Calibration] Optimal temperature T={T:.4f}")
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    str,
    epoch:     int,
    total:     int,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(imgs)

        if batch_idx % 20 == 0:
            logger.info(
                f"  Epoch {epoch}/{total}  batch {batch_idx}/{len(loader)}  "
                f"loss={loss.item():.4f}"
            )

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> Tuple[float, float, float, float]:
    """
    Validation pass.
    Returns (avg_loss, accuracy, avg_real_prob, avg_fake_prob).
    """
    model.eval()
    total_loss  = 0.0
    correct     = 0
    n           = 0
    real_probs: list[float] = []
    fake_probs: list[float] = []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)[:, 1]   # P(fake)

        total_loss += loss.item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(imgs)

        for p, lbl in zip(probs.cpu().tolist(), labels.cpu().tolist()):
            if lbl == 0:
                real_probs.append(p)
            else:
                fake_probs.append(p)

    avg_real = float(np.mean(real_probs)) if real_probs else 0.0
    avg_fake = float(np.mean(fake_probs)) if fake_probs else 0.0
    return total_loss / n, correct / n, avg_real, avg_fake


# ─────────────────────────────────────────────────────────────────────────────
# Data split utility
# ─────────────────────────────────────────────────────────────────────────────

def split_paths(
    paths:     list[Path],
    val_ratio: float = 0.20,
    seed:      int   = 42,
) -> Tuple[list[Path], list[Path]]:
    """Reproducible 80/20 train/val split."""
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - val_ratio))
    return shuffled[:cut], shuffled[cut:]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepShield EfficientNet-B0 deepfake detector")
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",      type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--val-split",  type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--real-dir",   type=str,   default=str(REAL_DIR))
    parser.add_argument("--fake-dir",   type=str,   default=str(FAKE_DIR))
    parser.add_argument("--output",     type=str,   default=str(OUTPUT_MODEL))
    parser.add_argument("--no-augment", action="store_true", default=False)
    parser.add_argument("--workers",    type=int,   default=0,
                        help="DataLoader num_workers (0=main thread, safe on Windows)")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect image paths ────────────────────────────────────────────────────
    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)

    real_paths = sorted(
        p for p in real_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    fake_paths = sorted(
        p for p in fake_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not real_paths or not fake_paths:
        logger.error(f"No images found!\n  real: {real_dir}\n  fake: {fake_dir}")
        return

    logger.info(f"Dataset: {len(real_paths)} real  {len(fake_paths)} fake")

    # ── Train / val split ──────────────────────────────────────────────────────
    train_real, val_real = split_paths(real_paths, args.val_split)
    train_fake, val_fake = split_paths(fake_paths, args.val_split)

    logger.info(
        f"Split: train={len(train_real)+len(train_fake)}  "
        f"val={len(val_real)+len(val_fake)}"
    )

    # ── Datasets + loaders ────────────────────────────────────────────────────
    train_ds = DeepfakeDataset(train_real, train_fake, augment=not args.no_augment)
    val_ds   = DeepfakeDataset(val_real,   val_fake,   augment=False)

    # Weighted sampler — balances real/fake even if counts differ
    labels_train  = [lbl for _, lbl in train_ds.samples]
    class_counts  = [labels_train.count(0), labels_train.count(1)]
    sample_weights = [1.0 / class_counts[lbl] for lbl in labels_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch,
        sampler     = sampler,
        num_workers = args.workers,
        pin_memory  = (device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch,
        shuffle     = False,
        num_workers = args.workers,
        pin_memory  = (device == "cuda"),
    )

    # ── Model + optimizer + loss ───────────────────────────────────────────────
    model     = build_model(device, pretrained=True)
    criterion = nn.CrossEntropyLoss()

    # Differential LR: backbone gets 10× lower LR than head
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params     = list(model.classifier.parameters())

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": args.lr / 10},
        {"params": head_params,     "lr": args.lr},
    ])

    # Cosine LR annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_epoch    = 0
    best_state    = None

    logger.info("=" * 60)
    logger.info(f"Training EfficientNet-B0  epochs={args.epochs}  lr={args.lr}  batch={args.batch}")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
        val_loss, val_acc, avg_real, avg_fake = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"avg_real_prob={avg_real:.3f}  avg_fake_prob={avg_fake:.3f}  "
            f"time={elapsed:.1f}s"
        )

        # ── Domain mismatch check ──────────────────────────────────────────────
        if avg_real > 0.50:
            logger.warning(
                f"  ⚠ Domain alert: avg P(fake|real)={avg_real:.3f} > 0.50 "
                f"— model may be overconfident on real faces"
            )
        if avg_fake < 0.50:
            logger.warning(
                f"  ⚠ Domain alert: avg P(fake|fake)={avg_fake:.3f} < 0.50 "
                f"— model may be under-detecting fakes"
            )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"  ✓ New best val_acc={best_val_acc:.4f} (epoch {best_epoch})")

    # ── Save best model ────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    output_path = Path(args.output)
    torch.save(model.state_dict(), str(output_path))
    logger.info(f"Model saved → {output_path}  (best epoch={best_epoch}  val_acc={best_val_acc:.4f})")

    # ── Temperature calibration ────────────────────────────────────────────────
    logger.info("Running temperature calibration on validation set...")
    T = calibrate_temperature(model, val_loader, device)

    temp_path = Path(str(output_path).replace(".pt", "_calibrated_T.txt"))
    temp_path.write_text(f"{T:.6f}\n")
    logger.info(f"Temperature T={T:.4f} saved → {temp_path}")

    # ── Final validation report ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL VALIDATION REPORT")
    logger.info("=" * 60)
    val_loss, val_acc, avg_real, avg_fake = eval_epoch(model, val_loader, criterion, device)
    logger.info(f"  Val accuracy    : {val_acc:.4f} ({val_acc*100:.1f}%)")
    logger.info(f"  Val loss        : {val_loss:.4f}")
    logger.info(f"  Avg P(fake|REAL): {avg_real:.4f}  {'✅ OK' if avg_real < 0.40 else '⚠ HIGH — domain mismatch risk'}")
    logger.info(f"  Avg P(fake|FAKE): {avg_fake:.4f}  {'✅ OK' if avg_fake > 0.70 else '⚠ LOW — model missing fakes'}")
    logger.info(f"  Temperature T   : {T:.4f}")
    logger.info("=" * 60)

    # Deployment decision
    if avg_real < 0.40 and avg_fake > 0.70:
        logger.info("✅ PASS — Model ready for integration testing")
        logger.info(f"   Checkpoint : {output_path}")
        logger.info(f"   Temperature: {temp_path}")
    else:
        logger.warning("⚠ FAIL — Check data distribution or re-train with more epochs")
        logger.warning("   DO NOT integrate into runtime until thresholds are met.")


if __name__ == "__main__":
    main()
