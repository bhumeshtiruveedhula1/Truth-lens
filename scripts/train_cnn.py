"""
scripts/train_cnn.py -- CNN Baseline: Real vs Non-Live Detection
================================================================
MobileNetV2 fine-tuned on data/cnn_dataset/{real,fake}/.
Visual signal only — fused with GRU later.

Usage:
    python scripts/train_cnn.py
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

# Windows console UTF-8 fix
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("data/cnn_dataset")
MODEL_OUT    = Path("models/cnn_baseline.pt")
CONFIG_OUT   = Path("models/cnn_config.json")
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

EPOCHS       = 15
BATCH        = 32
LR           = 1e-4
DROPOUT      = 0.25
EARLY_STOP   = 3          # patience on val_loss
VAL_SPLIT    = 0.20
SEED         = 42
THR_RANGE    = np.arange(0.30, 0.71, 0.05)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SEP  = "-" * 64
SEP2 = "=" * 64

torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """Label: real=0  fake=1"""
    def __init__(self, files: list[Path], labels: list[int], transform):
        self.files     = files
        self.labels    = labels
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.float32)


def _collect(root: Path):
    files, labels = [], []
    for label_name, label_id in (("real", 0), ("fake", 1)):
        d = root / label_name
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jpg")):
            files.append(p)
            labels.append(label_id)
    return files, labels


train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    m = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(in_f, 1),
    )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    return dict(
        acc  = accuracy_score(y_true, preds),
        prec = precision_score(y_true, preds, zero_division=0),
        rec  = recall_score(y_true, preds, zero_division=0),
        f1   = f1_score(y_true, preds, zero_division=0),
    )


def _sweep(y_true, probs):
    best_thr, best_f1 = 0.5, -1.0
    for t in THR_RANGE:
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr, best_f1


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, lbls)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(imgs)
            all_probs.extend(torch.sigmoid(logits).cpu().detach().numpy())
            all_labels.extend(lbls.cpu().numpy())
    n     = len(loader.dataset)
    probs = np.array(all_probs)
    trues = np.array(all_labels, dtype=int)
    m     = _metrics(trues, probs, 0.5)
    return total_loss / n, probs, trues, m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(SEP2)
    print("  CNN TRAINING: Real vs Non-Live Detection")
    print(SEP2)

    # ── STEP 1: Validate & load data ─────────────────────────────────────────
    print(f"\n{'='*56}")
    print("STEP 1 — DATASET VALIDATION")
    print(f"{'='*56}")

    if not DATASET_ROOT.exists():
        print("ERROR: data/cnn_dataset/ not found. Run build_cnn_dataset.py first.")
        sys.exit(1)

    files, labels = _collect(DATASET_ROOT)
    n_real = labels.count(0); n_fake = labels.count(1)
    total  = len(files)

    if total == 0:
        print("ERROR: No images found."); sys.exit(1)
    if n_real == 0 or n_fake == 0:
        print(f"ERROR: Need both classes. real={n_real} fake={n_fake}"); sys.exit(1)

    print(f"  Real images  : {n_real:,}")
    print(f"  Fake images  : {n_fake:,}")
    print(f"  Total        : {total:,}")
    print(f"  Ratio        : {n_real/total*100:.0f}% / {n_fake/total*100:.0f}%")

    # Brightness stats (sample 200 per class for speed)
    import cv2
    def _brightness(label_id, n=200):
        idx = [i for i,l in enumerate(labels) if l == label_id]
        np.random.shuffle(idx); means = []
        for i in idx[:n]:
            img = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
            if img is not None: means.append(img.mean())
        return float(np.mean(means)), float(np.std(means))

    r_mn, r_sd = _brightness(0); f_mn, f_sd = _brightness(1)
    print(f"\n  Brightness real  : {r_mn:.1f} ± {r_sd:.1f}")
    print(f"  Brightness fake  : {f_mn:.1f} ± {f_sd:.1f}")
    print(f"  Brightness gap   : {abs(r_mn-f_mn):.1f} pts", end="")
    bright_bias = abs(r_mn - f_mn) > 25
    print("  !! WARN: brightness bias" if bright_bias else "  OK")

    # Shape check (sample 10)
    bad_shapes = 0
    for p in files[:10]:
        img = cv2.imread(str(p))
        if img is not None and img.shape[:2] != (224, 224): bad_shapes += 1
    print(f"  Shape (224x224)  : {'OK' if bad_shapes == 0 else f'{bad_shapes} errors in first 10!'}")

    print(f"\n  Samples real : {files[0].name}")
    print(f"  Samples fake : {next(f for f,l in zip(files,labels) if l==1).name}")

    imbalance = n_real/total > 0.70 or n_fake/total > 0.70
    if imbalance:
        print(f"\n  !! WARN: class imbalance > 70/30")

    # ── STEP 2: Split ────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("STEP 2 — TRAIN/VAL SPLIT")
    print(f"{'='*56}")

    full_ds = FaceDataset(files, labels, train_tf)
    n_val   = int(total * VAL_SPLIT)
    n_train = total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    # Val uses val transforms — wrap with override
    class _ValWrap(Dataset):
        def __init__(self, subset):
            self.subset = subset
        def __len__(self): return len(self.subset)
        def __getitem__(self, i):
            img_t, lbl = self.subset[i]
            # Re-read original file with val_tf
            idx  = self.subset.indices[i]
            img  = Image.open(files[idx]).convert("RGB")
            return val_tf(img), lbl

    val_ds_clean = _ValWrap(val_ds)
    train_lbl = [labels[i] for i in train_ds.indices]
    val_lbl   = [labels[i] for i in val_ds.indices]
    print(f"  Train : {n_train:,}  real={train_lbl.count(0):,}  fake={train_lbl.count(1):,}")
    print(f"  Val   : {n_val:,}  real={val_lbl.count(0):,}  fake={val_lbl.count(1):,}")

    train_dl = DataLoader(train_ds,     batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds_clean, batch_size=BATCH, shuffle=False, num_workers=0)

    # ── STEP 3: Model ────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("STEP 3 — MODEL")
    print(f"{'='*56}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device     : {device}")
    model     = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"  Backbone   : MobileNetV2 (pretrained ImageNet)")
    print(f"  Classifier : Dropout({DROPOUT}) → Linear(1280→1)")
    print(f"  Loss       : BCEWithLogitsLoss")
    print(f"  Optimizer  : Adam lr={LR}")
    print(f"  Epochs     : {EPOCHS}  batch={BATCH}  early_stop={EARLY_STOP}")

    # ── STEP 4: Training loop ────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("STEP 4 — TRAINING")
    print(f"{'='*56}")
    print(f"  {'Ep':>3}  {'TrLoss':>8}  {'VaLoss':>8}  "
          f"{'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*58}")

    best_f1 = -1.0; best_state = None; best_epoch = 0
    no_improve = 0; best_val_loss = float("inf")
    history = []
    val_probs_best = val_labels_best = None

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, _, _, tr_m       = _run_epoch(model, train_dl, criterion, optimizer, device, True)
        va_loss, va_probs, va_lbl, va_m = _run_epoch(model, val_dl, criterion, optimizer, device, False)

        mark = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss; no_improve = 0
            if va_m["f1"] >= best_f1:
                best_f1, best_epoch = va_m["f1"], ep
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                val_probs_best, val_labels_best = va_probs.copy(), va_lbl.copy()
                mark = " *"
        else:
            no_improve += 1

        # Overfitting check
        gap_acc = tr_m["acc"] - va_m["acc"]
        of_warn = " !! OVERFIT" if (tr_loss < va_loss * 0.5 and gap_acc > 0.15) else ""

        print(f"  {ep:>3}  {tr_loss:>8.4f}  {va_loss:>8.4f}  "
              f"{va_m['acc']:>6.3f}  {va_m['prec']:>6.3f}  "
              f"{va_m['rec']:>6.3f}  {va_m['f1']:>6.3f}"
              f"{mark}{of_warn}")

        history.append(dict(ep=ep, tr_loss=tr_loss, va_loss=va_loss, **va_m))

        if no_improve >= EARLY_STOP:
            print(f"\n  Early stop at epoch {ep} (no val_loss improvement for {EARLY_STOP} epochs)")
            break

    model.load_state_dict(best_state)
    print(f"\n  Best epoch: {best_epoch}  val_F1={best_f1:.4f}")

    # ── STEP 5: Threshold sweep ──────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("STEP 5 — THRESHOLD SWEEP")
    print(f"{'='*56}")
    print(f"  {'Thr':>5}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*40}")
    for t in THR_RANGE:
        m = _metrics(val_labels_best, val_probs_best, t)
        print(f"  {t:>5.2f}  {m['acc']:>6.3f}  {m['prec']:>6.3f}  {m['rec']:>6.3f}  {m['f1']:>6.3f}")

    best_thr, best_thr_f1 = _sweep(val_labels_best, val_probs_best)
    final_m = _metrics(val_labels_best, val_probs_best, best_thr)
    print(f"\n  Best threshold : {best_thr:.2f}  F1={best_thr_f1:.4f}")

    # ── STEP 6: Confusion matrix ─────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("STEP 6 — CONFUSION MATRIX (at best threshold)")
    print(f"{'='*56}")
    preds_best = (val_probs_best >= best_thr).astype(int)
    cm = confusion_matrix(val_labels_best, preds_best)
    tn, fp, fn, tp = cm.ravel()
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  {'':20}  Pred REAL  Pred FAKE")
    print(f"  {'Actual REAL':20}  {tn:>9}  {fp:>9}")
    print(f"  {'Actual FAKE':20}  {fn:>9}  {tp:>9}")
    print(f"\n  Real accuracy (specificity) : {real_acc:.3f}")
    print(f"  Fake accuracy (recall)      : {fake_acc:.3f}")

    # ── Overfitting summary ──────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("STEP 7 — OVERFITTING CHECK")
    print(f"{'='*56}")
    final_h = history[best_epoch - 1]
    gap = final_h["acc"] - final_m["acc"]
    print(f"  Train loss (best epoch) : {final_h['tr_loss']:.4f}")
    print(f"  Val   loss (best epoch) : {final_h['va_loss']:.4f}")
    print(f"  Train/val accuracy gap  : {gap:.3f}", end="")
    if gap > 0.10:
        print("  !! Possible overfitting")
    else:
        print("  OK")

    # ── STEP 7: Save ────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("STEP 8 — SAVE")
    print(f"{'='*56}")
    ckpt = dict(
        model_state      = model.state_dict(),
        best_threshold   = best_thr,
        imagenet_mean    = IMAGENET_MEAN,
        imagenet_std     = IMAGENET_STD,
        input_size       = 224,
        val_f1           = round(best_thr_f1, 4),
        val_real_acc     = round(real_acc, 4),
        val_fake_acc     = round(fake_acc, 4),
    )
    torch.save(ckpt, MODEL_OUT)
    cfg = dict(
        input_size     = 224,
        imagenet_mean  = IMAGENET_MEAN,
        imagenet_std   = IMAGENET_STD,
        best_threshold = best_thr,
        val_f1         = round(best_thr_f1, 4),
        val_real_acc   = round(real_acc, 4),
        val_fake_acc   = round(fake_acc, 4),
        label_map      = {"0": "real", "1": "fake"},
    )
    CONFIG_OUT.write_text(json.dumps(cfg, indent=2))
    print(f"  Model   : {MODEL_OUT}  ({MODEL_OUT.stat().st_size/1024:.0f} KB)")
    print(f"  Config  : {CONFIG_OUT}")

    # ── STEP 8: Final interpretation ────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  FINAL INTERPRETATION")
    print(SEP2)
    reliable = best_thr_f1 >= 0.75 and fake_acc >= 0.70 and real_acc >= 0.70
    print(f"  1. Is model reliable?              {'YES' if reliable else 'NO — F1 or accuracy below target'}")
    print(f"     F1={best_thr_f1:.3f}  real_acc={real_acc:.3f}  fake_acc={fake_acc:.3f}")

    print(f"\n  2. Is fake recall strong enough?   {'YES' if fake_acc >= 0.75 else 'NO — fake detection too weak'}")
    print(f"     Fake recall = {fake_acc:.3f}  (target >= 0.75)")

    print(f"\n  3. Brightness bias detected?       {'YES — CNN may be using brightness as shortcut' if bright_bias else 'NO — OK'}")
    if bright_bias:
        print(f"     real_mean={r_mn:.0f}  fake_mean={f_mn:.0f}  gap={abs(r_mn-f_mn):.0f}")
        print(f"     Action: add more varied fake sessions or apply brightness normalization")

    ready = reliable and fake_acc >= 0.75
    print(f"\n  4. Ready for real-time integration? {'YES' if ready else 'NO — retrain or collect more data'}")
    if ready:
        print(f"     Load models/cnn_baseline.pt in inference layer")
        print(f"     Use threshold={best_thr:.2f} for binary decision")
        print(f"     Fuse with GRU: CNN_score * 0.4 + GRU_score * 0.6 (suggested)")
    print(SEP2)


if __name__ == "__main__":
    main()
