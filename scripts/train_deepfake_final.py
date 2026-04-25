"""
scripts/train_deepfake_final.py
================================
Domain-aligned EfficientNet-B0 deepfake detector.
Trained on data/deepfake_final/ (webcam + Celeb-DF + OBS).

Key design choices (anti-domain-mismatch):
  - ImageNet normalize + JPEG compression augmentation
  - BCEWithLogitsLoss with pos_weight to prevent SAFE bias
  - Differential LR: backbone 1e-5, head 1e-4
  - Validation checks: avg P(fake|real) < 0.40, P(fake|fake) > 0.70

Usage:
  python scripts/train_deepfake_final.py
  python scripts/train_deepfake_final.py --epochs 15 --batch 32
  python scripts/train_deepfake_final.py --batch 24   # if OOM
"""
from __future__ import annotations

import argparse
import io
import logging
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
log = logging.getLogger("train_deepfake")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
TRAIN_REAL = ROOT / "data/deepfake_final/train/real"
TRAIN_FAKE = ROOT / "data/deepfake_final/train/fake"
VAL_REAL   = ROOT / "data/deepfake_final/val/real"
VAL_FAKE   = ROOT / "data/deepfake_final/val/fake"
OUTPUT     = ROOT / "models/deepfake_efficientnet_final.pt"
LOG_CSV    = ROOT / "models/deepfake_training_log.csv"

IMG_SIZE   = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── GPU Setup (Task 1) ────────────────────────────────────────────────────────

def setup_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {name}  VRAM={vram:.1f}GB  cuDNN.benchmark=True")
    else:
        device = torch.device("cpu")
        log.warning("CUDA not available -- training on CPU (slow)")
    return device


# ── Dataset (Task 4: Augmentation) ───────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Binary face-crop dataset.
      label 0 = REAL
      label 1 = FAKE

    Train augmentation:
      RandomHorizontalFlip + ColorJitter + GaussianBlur +
      JPEG recompression (40-95) + Gaussian noise
    These force the model to learn facial realism, not compression artifacts.
    """

    def __init__(self, real_paths: list[Path], fake_paths: list[Path],
                 augment: bool = True) -> None:
        self.samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        random.shuffle(self.samples)
        self.augment = augment

        self._aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.35, contrast=0.35,
                                   saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.04),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.8)),
        ])
        self._base = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

        if self.augment:
            img = self._aug(img)
            img = self._jpeg_compress(img)
            img = self._add_noise(img)

        return self._base(img), label

    @staticmethod
    def _jpeg_compress(img: Image.Image) -> Image.Image:
        """Simulate webcam JPEG compression at random quality 40-95."""
        q = random.randint(40, 95)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).copy()

    @staticmethod
    def _add_noise(img: Image.Image) -> Image.Image:
        """Gaussian sensor noise (sigma 0-10)."""
        arr   = np.asarray(img, dtype=np.float32)
        sigma = random.uniform(0.0, 10.0)
        noise = np.random.randn(*arr.shape).astype(np.float32) * sigma
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def make_loader(real_dir: Path, fake_dir: Path, batch: int,
                workers: int, augment: bool, device: torch.device) -> DataLoader:
    real_paths = sorted(p for p in real_dir.rglob("*.jpg"))
    fake_paths = sorted(p for p in fake_dir.rglob("*.jpg"))

    if not real_paths or not fake_paths:
        raise FileNotFoundError(f"No images in {real_dir} or {fake_dir}")

    ds = DeepfakeDataset(real_paths, fake_paths, augment=augment)

    # Weighted sampler for perfect 50/50 batches (Task 5)
    labels = [lbl for _, lbl in ds.samples]
    counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)

    pin = (device.type == "cuda")
    return DataLoader(
        ds, batch_size=batch, sampler=sampler,
        num_workers=workers, pin_memory=pin,
        persistent_workers=(workers > 0),
    )


# ── Model (Task 2) ────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f  = model.classifier[1].in_features            # 1280
    model.classifier[1] = nn.Linear(in_f, 1)           # single logit for BCEWithLogitsLoss
    model = model.to(device)
    log.info(f"EfficientNet-B0: head=Linear({in_f}->1)  device={device}")
    return model


def make_optimizer(model: nn.Module, lr_head: float, lr_backbone: float):
    backbone_p = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_p     = list(model.classifier.parameters())
    return torch.optim.Adam([
        {"params": backbone_p, "lr": lr_backbone},
        {"params": head_p,     "lr": lr_head},
    ])


# ── Training loop (Task 6) ────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = correct = n = 0
    for i, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)
        correct    += ((logits.sigmoid() >= 0.5).float() == labels).sum().item()
        n          += len(imgs)

        if i % 20 == 0:
            log.info(f"  E{epoch}/{total_epochs} batch {i}/{len(loader)}"
                     f"  loss={loss.item():.4f}"
                     f"  gpu_mb={torch.cuda.memory_allocated()//1e6:.0f}"
                     if device.type == "cuda" else
                     f"  E{epoch}/{total_epochs} batch {i}/{len(loader)}"
                     f"  loss={loss.item():.4f}")

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = n = 0
    real_probs: list[float] = []
    fake_probs: list[float] = []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = logits.sigmoid()

        total_loss += loss.item() * len(imgs)
        correct    += ((probs >= 0.5).float() == labels).sum().item()
        n          += len(imgs)

        for p, lbl in zip(probs.cpu().squeeze(1).tolist(), labels.cpu().squeeze(1).tolist()):
            (real_probs if lbl == 0 else fake_probs).append(p)

    avg_real = float(np.mean(real_probs)) if real_probs else 0.0
    avg_fake = float(np.mean(fake_probs)) if fake_probs else 0.0
    return total_loss / n, correct / n, avg_real, avg_fake


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",   type=int,   default=15)
    p.add_argument("--batch",    type=int,   default=32,   help="32 normal; 24 if OOM")
    p.add_argument("--lr",       type=float, default=1e-4, help="head LR")
    p.add_argument("--workers",  type=int,   default=2,    help="DataLoader workers")
    p.add_argument("--train-real", default=str(TRAIN_REAL))
    p.add_argument("--train-fake", default=str(TRAIN_FAKE))
    p.add_argument("--val-real",   default=str(VAL_REAL))
    p.add_argument("--val-fake",   default=str(VAL_FAKE))
    p.add_argument("--output",     default=str(OUTPUT))
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = setup_device()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    log.info("Building data loaders...")
    train_loader = make_loader(
        Path(args.train_real), Path(args.train_fake),
        args.batch, args.workers, augment=True, device=device,
    )
    val_loader = make_loader(
        Path(args.val_real), Path(args.val_fake),
        args.batch, args.workers, augment=False, device=device,
    )
    log.info(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # Model
    model     = build_model(device)
    optimizer = make_optimizer(model, lr_head=args.lr, lr_backbone=args.lr / 10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Task 3: BCEWithLogitsLoss with pos_weight
    n_real = len(list(Path(args.train_real).rglob("*.jpg")))
    n_fake = len(list(Path(args.train_fake).rglob("*.jpg")))
    pos_w  = torch.tensor([n_real / n_fake], device=device)
    log.info(f"pos_weight = {pos_w.item():.3f}  (real={n_real} fake={n_fake})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # Training loop
    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 0
    csv_rows      = ["epoch,train_loss,val_loss,val_acc,avg_real,avg_fake,time_s"]

    log.info("=" * 60)
    log.info(f"Training EfficientNet-B0  epochs={args.epochs}  batch={args.batch}"
             f"  device={device}")
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
        val_loss, val_acc, avg_real, avg_fake = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train={train_loss:.4f}/{train_acc:.3f}  "
            f"val={val_loss:.4f}/{val_acc:.3f}  "
            f"real_prob={avg_real:.3f}  fake_prob={avg_fake:.3f}  "
            f"time={elapsed:.0f}s"
        )

        # Domain mismatch alerts
        if avg_real > 0.50:
            log.warning(f"  !! Domain alert: avg P(fake|real)={avg_real:.3f} > 0.50")
        if avg_fake < 0.50:
            log.warning(f"  !! Domain alert: avg P(fake|fake)={avg_fake:.3f} < 0.50")

        csv_rows.append(
            f"{epoch},{train_loss:.5f},{val_loss:.5f},"
            f"{val_acc:.4f},{avg_real:.4f},{avg_fake:.4f},{elapsed:.1f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.info(f"  >> New best val_loss={best_val_loss:.4f} (epoch {best_epoch})")

        # VRAM log
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated() / 1e6
            resvd = torch.cuda.memory_reserved()  / 1e6
            log.info(f"  GPU: allocated={alloc:.0f}MB  reserved={resvd:.0f}MB")

    # Restore best and save
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), str(args.output))
    log.info(f"Model saved -> {args.output}  (best epoch={best_epoch})")

    # Write CSV log
    LOG_CSV.write_text("\n".join(csv_rows))
    log.info(f"Training log -> {LOG_CSV}")

    # Task 8: Final validation report
    log.info("\n" + "=" * 60)
    log.info("TASK 8 -- FINAL VALIDATION CHECK")
    log.info("=" * 60)
    _, val_acc, avg_real, avg_fake = eval_epoch(model, val_loader, criterion, device)
    log.info(f"  Val accuracy    : {val_acc:.4f}")
    log.info(f"  avg P(fake|REAL): {avg_real:.4f}  "
             f"{'PASS (<0.40)' if avg_real < 0.40 else 'FAIL (>0.40) -- domain mismatch risk'}")
    log.info(f"  avg P(fake|FAKE): {avg_fake:.4f}  "
             f"{'PASS (>0.70)' if avg_fake > 0.70 else 'FAIL (<0.70) -- model missing fakes'}")

    log.info("=" * 60)
    if avg_real < 0.40 and avg_fake > 0.70:
        log.info("  DEPLOYMENT VERDICT: PASS -- ready for integration")
    else:
        log.warning("  DEPLOYMENT VERDICT: FAIL -- DO NOT integrate yet")
        log.warning("  Review training logs and dataset distribution.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
