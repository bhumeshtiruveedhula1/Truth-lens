"""
train_refined.py  --  GRU Refinement Pass (DeepShield Liveness)
================================================================
Refinement-only pass on top of the existing GRU baseline.
Identical architecture, identical dataset, identical split logic.

What changed vs train.py (baseline):
  1. Loss: Focal Loss (gamma=2.0) instead of BCEWithLogitsLoss
       -> down-weights easy negatives, forces harder fake samples
          to dominate the gradient -> improves recall
  2. LR:  5e-4 (from 1e-3)  + ReduceLROnPlateau(F1, patience=3)
  3. Early stopping: on val F1 (not val loss) -- directly optimises target
  4. Epochs: 30 max
  5. Threshold sweep [0.30..0.70 step 0.05] run on val set after every
     epoch; best F1 threshold saved into checkpoint

Architecture, features, dataset, split: UNCHANGED.

Usage
-----
  python train_refined.py
  python train_refined.py --epochs 30 --lr 5e-4 --gamma 2.0 --patience 5
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    sys.exit("ERROR: PyTorch not found.\n"
             "  pip install torch --index-url https://download.pytorch.org/whl/cpu")

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )
except ImportError:
    sys.exit("ERROR: scikit-learn not found.  pip install scikit-learn")


# ===========================================================================
# 1.  Model  (UNCHANGED from baseline)
# ===========================================================================

class GRUClassifier(nn.Module):
    """Single-layer GRU -> last hidden state -> binary logit.  Unchanged."""

    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)           # h: (1, B, H)
        return self.head(h.squeeze(0)).squeeze(-1)   # (B,)


# ===========================================================================
# 2.  Focal Loss
# ===========================================================================

class FocalLoss(nn.Module):
    """Binary Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2.0: standard recommendation (Lin et al., 2017).
    Uses logits as input (numerically stable via sigmoid + log-sigmoid).
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE term per sample
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
        )
        # p_t = probability of the true class
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ===========================================================================
# 3.  Data loading & split  (UNCHANGED from baseline)
# ===========================================================================

def load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    required = {"X", "y", "groups", "session_ids", "features", "feat_mean", "feat_std"}
    missing = required - set(d.keys())
    if missing:
        sys.exit(
            f"ERROR: {path} is missing keys: {missing}\n"
            "Rebuild with scripts/build_sequences.py (latest version)."
        )
    return d


def group_split_indices(
    groups: np.ndarray,
    session_ids: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Stratified group split -- no leakage.  Identical to baseline."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)

    real_groups, fake_groups = [], []
    for g in unique_groups:
        mask     = groups == g
        majority = int(np.round(y[mask].mean()))
        (fake_groups if majority == 1 else real_groups).append(g)

    rng.shuffle(real_groups)
    rng.shuffle(fake_groups)

    n_val_real = max(1, math.ceil(len(real_groups) * val_frac))
    n_val_fake = max(1, math.ceil(len(fake_groups) * val_frac))

    val_groups   = real_groups[:n_val_real] + fake_groups[:n_val_fake]
    train_groups = real_groups[n_val_real:]  + fake_groups[n_val_fake:]

    train_idx = np.where(np.isin(groups, train_groups))[0]
    val_idx   = np.where(np.isin(groups, val_groups))[0]

    train_sids = [str(session_ids[g]) for g in sorted(train_groups)]
    val_sids   = [str(session_ids[g]) for g in sorted(val_groups)]

    return train_idx, val_idx, train_sids, val_sids


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y.astype(np.float32))
    train_ds = TensorDataset(X_t[train_idx], y_t[train_idx])
    val_ds   = TensorDataset(X_t[val_idx],   y_t[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


# ===========================================================================
# 4.  Training helpers
# ===========================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Single pass.  Returns (loss, y_true, raw_probs_float)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_true, all_probs = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)
            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            all_true.extend(yb.long().cpu().numpy().tolist())
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_true), np.array(all_probs, dtype=np.float32)


def threshold_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    preds = (probs >= thr).astype(int)
    return {
        "thr":       thr,
        "acc":       accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall":    recall_score(y_true, preds, zero_division=0),
        "f1":        f1_score(y_true, preds, zero_division=0),
        "cm":        confusion_matrix(y_true, preds),
        "preds":     preds,
    }


def threshold_sweep(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float],
) -> tuple[dict, list[dict]]:
    """Sweep thresholds; return (best_by_f1, all_results)."""
    results = [threshold_metrics(y_true, probs, t) for t in thresholds]
    best    = max(results, key=lambda r: r["f1"])
    return best, results


# ===========================================================================
# 5.  Reporting helpers
# ===========================================================================

def print_metrics_table(results: list[dict]) -> None:
    print(f"\n  {'Threshold':>10}  {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
    print("  " + "-" * 46)
    for r in results:
        print(f"  {r['thr']:>10.2f}  {r['acc']:>7.4f}  {r['precision']:>7.4f}"
              f"  {r['recall']:>7.4f}  {r['f1']:>7.4f}")


def print_cm(cm: np.ndarray, title: str = "") -> None:
    if title:
        print(f"\n  {title}")
    print(f"               pred_real  pred_fake")
    for i, lbl in enumerate(["real (0)", "fake (1)"]):
        print(f"  true_{lbl}  {cm[i, 0]:>9}  {cm[i, 1]:>9}")


def save_val_predictions(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    best_thr: float,
    out_path: Path,
) -> None:
    model.eval()
    all_probs: list[float] = []
    for start in range(0, len(val_idx), 256):
        idx = val_idx[start : start + 256]
        xb  = torch.from_numpy(X[idx]).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(xb)).cpu().numpy()
        all_probs.extend(probs.tolist())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "true_label", "pred_label", "prob_fake"])
        for i, si in enumerate(val_idx):
            pred = 1 if all_probs[i] >= best_thr else 0
            w.writerow([si, int(y[si]), pred, f"{all_probs[i]:.6f}"])
    print(f"\n  Val predictions saved -> {out_path}")


def show_sample_predictions(
    probs: np.ndarray,
    y_true: np.ndarray,
    val_idx: np.ndarray,
    best_thr: float,
    n: int = 10,
) -> None:
    lbl = lambda v: "fake" if v == 1 else "real"
    preds = (probs >= best_thr).astype(int)
    print(f"\n  Sample predictions (first {n} val windows, thr={best_thr:.2f}):")
    print(f"  {'idx':>6}  {'true':>6}  {'pred':>6}  {'prob':>8}  {'ok?':>4}")
    print("  " + "-" * 42)
    for i in range(min(n, len(val_idx))):
        si = val_idx[i]
        gt = int(y_true[i])
        pr = preds[i]
        pb = probs[i]
        ok = "OK" if gt == pr else "MISS"
        print(f"  {si:>6}  {lbl(gt):>6}  {lbl(pr):>6}  {pb:>8.4f}  {ok:>4}")


# ===========================================================================
# 6.  Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRU Refinement Pass -- Focal Loss + F1-based early stopping"
    )
    parser.add_argument("--data",      type=Path,  default=Path("data/sequences.npz"))
    parser.add_argument("--out",       type=Path,  default=Path("models/gru_refined.pt"))
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=5e-4)
    parser.add_argument("--hidden",    type=int,   default=128)
    parser.add_argument("--gamma",     type=float, default=2.0,
                        help="Focal loss gamma (default 2.0)")
    parser.add_argument("--pos-weight",type=float, default=1.0,
                        help="Focal loss pos_weight for fake class (>1 biases recall)")
    parser.add_argument("--patience",  type=int,   default=5,
                        help="Early-stop patience on val F1")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--val-preds", type=Path,  default=Path("data/val_predictions_refined.csv"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Config: lr={args.lr}  gamma={args.gamma}  pos_weight={args.pos_weight}"
          f"  patience={args.patience}  epochs={args.epochs}")

    # ------------------------------------------------------------------
    # Load & split
    # ------------------------------------------------------------------
    print(f"\nLoading {args.data} ...")
    d = load_npz(args.data)

    X           = d["X"]
    y           = d["y"]
    groups      = d["groups"]
    session_ids = d["session_ids"]
    features    = list(d["features"])

    N, T, F = X.shape
    print(f"  Dataset : {N:,} samples  |  window={T}  |  F={F}")
    print(f"  Labels  : real={int((y==0).sum()):,}  fake={int((y==1).sum()):,}")

    train_idx, val_idx, train_sids, val_sids = group_split_indices(
        groups, session_ids, y, val_frac=0.20, seed=args.seed
    )

    # Leakage guard
    assert len(set(train_sids) & set(val_sids)) == 0, \
        "LEAKAGE DETECTED: overlapping sessions in train and val!"

    print(f"\n  Train sessions ({len(train_sids)}): {[s[-8:] for s in train_sids]}")
    print(f"  Val   sessions ({len(val_sids)}):   {[s[-8:] for s in val_sids]}")
    print(f"  Train samples : {len(train_idx):,}  |  Val samples: {len(val_idx):,}")
    print(f"  Train real/fake: {int((y[train_idx]==0).sum())}/{int((y[train_idx]==1).sum())}")
    print(f"  Val   real/fake: {int((y[val_idx]==0).sum())}/{int((y[val_idx]==1).sum())}")

    train_loader, val_loader = make_loaders(X, y, train_idx, val_idx, args.batch)

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model     = GRUClassifier(input_size=F, hidden_size=args.hidden).to(device)
    criterion = FocalLoss(gamma=args.gamma, pos_weight=args.pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {total_params:,}")
    print(f"  Loss: FocalLoss(gamma={args.gamma}, pos_weight={args.pos_weight})")

    # Threshold candidates (0.30 → 0.70, step 0.05)
    thresholds = [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]

    # ------------------------------------------------------------------
    # Training loop — early stop on val F1 at best threshold
    # ------------------------------------------------------------------
    best_val_f1    = -1.0
    best_thr       = 0.50
    patience_count = 0
    best_epoch     = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>10}  "
          f"{'F1@best':>8}  {'Thr':>5}  {'Rec':>7}  {'Prec':>7}  LR")
    print(f"{'='*72}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, _, _         = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,  y_true, probs = run_epoch(model, val_loader,   criterion, None,      device)

        # Threshold sweep on val probs
        best_result, _ = threshold_sweep(y_true, probs, thresholds)

        epoch_f1  = best_result["f1"]
        epoch_thr = best_result["thr"]
        epoch_rec = best_result["recall"]
        epoch_pre = best_result["precision"]

        # LR scheduler step (maximise F1)
        scheduler.step(epoch_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{epoch_f1:>8.4f}  {epoch_thr:>5.2f}  {epoch_rec:>7.4f}"
              f"  {epoch_pre:>7.4f}  {current_lr:.2e}  [{elapsed:.1f}s]")

        # Save checkpoint if val F1 improved
        if epoch_f1 > best_val_f1:
            best_val_f1    = epoch_f1
            best_thr       = epoch_thr
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_loss":     val_loss,
                "val_f1":       epoch_f1,
                "best_threshold": best_thr,
                "features":     features,
                "hidden_size":  args.hidden,
                "input_size":   F,
                "gamma":        args.gamma,
                "pos_weight":   args.pos_weight,
            }, args.out)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  [Early stop] No F1 improvement for {args.patience} epochs. "
                      f"Best epoch: {best_epoch}  (F1={best_val_f1:.4f} @ thr={best_thr:.2f})")
                break

    # ------------------------------------------------------------------
    # Final evaluation — load best checkpoint
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  Loading best checkpoint (epoch {best_epoch}) ...")
    ckpt = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    saved_thr = ckpt["best_threshold"]

    _, y_true, probs = run_epoch(model, val_loader, criterion, None, device)

    # --- Full threshold sweep table ---
    best_result, all_results = threshold_sweep(y_true, probs, thresholds)

    print(f"\n  --- THRESHOLD SWEEP (val set, best epoch {best_epoch}) ---")
    print_metrics_table(all_results)

    print(f"\n  Best threshold: {saved_thr:.2f}  (maximises F1)")

    # --- Metrics at saved threshold ---
    m_best = threshold_metrics(y_true, probs, saved_thr)
    print(f"\n  --- FINAL VALIDATION METRICS @ thr={saved_thr:.2f} ---")
    print(f"  Accuracy  : {m_best['acc']:.4f}  ({m_best['acc']*100:.1f}%)")
    print(f"  Precision : {m_best['precision']:.4f}")
    print(f"  Recall    : {m_best['recall']:.4f}")
    print(f"  F1        : {m_best['f1']:.4f}")
    print_cm(m_best["cm"], title="Confusion matrix (rows=true, cols=pred):")

    # --- Comparison vs baseline (hardcoded baseline numbers) ---
    print(f"\n  --- COMPARISON vs BASELINE (thr=0.50) ---")
    m_at_50 = threshold_metrics(y_true, probs, 0.50)
    print(f"  {'Metric':<12}  {'Baseline':>10}  {'Refined':>10}  {'Delta':>10}")
    print("  " + "-" * 48)
    baseline = {"Accuracy": 0.6113, "Precision": 0.7406, "Recall": 0.4189, "F1": 0.5351}
    refined  = {
        "Accuracy":  m_at_50["acc"],
        "Precision": m_at_50["precision"],
        "Recall":    m_at_50["recall"],
        "F1":        m_at_50["f1"],
    }
    for k in baseline:
        delta = refined[k] - baseline[k]
        sign  = "+" if delta >= 0 else ""
        print(f"  {k:<12}  {baseline[k]:>10.4f}  {refined[k]:>10.4f}  {sign}{delta:>9.4f}")

    print(f"\n  --- BEST THRESHOLD GAINS vs BASELINE ---")
    print(f"  {'Metric':<12}  {'Baseline':>10}  {'@thr={:.2f}'.format(saved_thr):>10}  {'Delta':>10}")
    print("  " + "-" * 48)
    refined_best = {
        "Accuracy":  m_best["acc"],
        "Precision": m_best["precision"],
        "Recall":    m_best["recall"],
        "F1":        m_best["f1"],
    }
    for k, bv in baseline.items():
        rv    = refined_best[k]
        delta = rv - bv
        sign  = "+" if delta >= 0 else ""
        print(f"  {k:<12}  {bv:>10.4f}  {rv:>10.4f}  {sign}{delta:>9.4f}")

    # --- Sample preds & CSV ---
    show_sample_predictions(probs, y_true, val_idx, saved_thr, n=10)
    save_val_predictions(model, X, y, val_idx, device, saved_thr, args.val_preds)

    print(f"\n  Refined model saved -> {args.out}")
    print(f"  Best threshold     -> {saved_thr:.2f}  (stored in checkpoint)")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
