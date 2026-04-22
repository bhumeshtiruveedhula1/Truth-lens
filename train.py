"""
train.py  --  GRU Baseline Trainer (DeepShield Liveness)
=========================================================
Trains a single-layer GRU classifier on sequences.npz produced by
scripts/build_sequences.py.

Split guarantee
---------------
  The train / validation split is performed on SESSION groups, not on
  individual windows.  No window from the same session can appear in
  both splits (zero data-leakage by construction).

Architecture
------------
  GRU(input=F, hidden=128, layers=1, batch_first=True)
  -> last hidden state -> Linear(128, 1)
  -> BCEWithLogitsLoss

Usage
-----
  python train.py
  python train.py --data data/sequences.npz --epochs 20 --batch 32 --lr 1e-3
"""

from __future__ import annotations

import argparse
import csv
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
    sys.exit("ERROR: PyTorch not found.  Run:\n"
             "  pip install torch --index-url https://download.pytorch.org/whl/cpu")

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )
except ImportError:
    sys.exit("ERROR: scikit-learn not found.  Run:  pip install scikit-learn")


# ===========================================================================
# 1.  Model
# ===========================================================================

class GRUClassifier(nn.Module):
    """Single-layer GRU -> last hidden state -> binary logit."""

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
        # x: (B, T, F)
        _, h = self.gru(x)           # h: (1, B, H)
        h_last = h.squeeze(0)        # (B, H)
        return self.head(h_last).squeeze(-1)   # (B,)


# ===========================================================================
# 2.  Data loading & split
# ===========================================================================

def load_npz(path: Path) -> dict:
    """Load sequences.npz; verify required keys are present."""
    d = np.load(path, allow_pickle=True)
    required = {"X", "y", "groups", "session_ids", "features", "feat_mean", "feat_std"}
    missing = required - set(d.keys())
    if missing:
        sys.exit(
            f"ERROR: {path} is missing keys: {missing}\n"
            "Rebuild with the latest scripts/build_sequences.py which emits "
            "'groups' and 'session_ids'."
        )
    return d


def group_split_indices(
    groups: np.ndarray,
    session_ids: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Stratified group split — no leakage.

    Sessions are partitioned by label class first (real / fake), then
    ~val_frac of each class's sessions are placed in val.  This prevents
    the pathological case where all val sessions are the same class.

    Returns
    -------
    train_idx, val_idx   : 1-D int arrays
    train_sids, val_sids : lists of UUID strings (for reporting)
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)

    # Determine majority label for each session
    real_groups, fake_groups = [], []
    for g in unique_groups:
        mask = groups == g
        majority = int(np.round(y[mask].mean()))  # 0=real, 1=fake
        (fake_groups if majority == 1 else real_groups).append(g)

    rng.shuffle(real_groups)
    rng.shuffle(fake_groups)

    # Take ceil(val_frac * n_class) sessions per class for val
    import math
    n_val_real = max(1, math.ceil(len(real_groups) * val_frac))
    n_val_fake = max(1, math.ceil(len(fake_groups) * val_frac))

    val_groups   = real_groups[:n_val_real] + fake_groups[:n_val_fake]
    train_groups = real_groups[n_val_real:] + fake_groups[n_val_fake:]

    train_mask = np.isin(groups, train_groups)
    val_mask   = np.isin(groups, val_groups)

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]

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
# 3.  Training loop
# ===========================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Single train or eval epoch.  Returns (loss, y_true, y_pred_binary)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_true, all_pred = [], []

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
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_true.extend(yb.long().cpu().numpy().tolist())
            all_pred.extend(preds.tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_true), np.array(all_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc":       accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "cm":        confusion_matrix(y_true, y_pred),
    }


# ===========================================================================
# 4.  Prediction sample display
# ===========================================================================

def show_sample_predictions(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    n: int = 8,
) -> None:
    model.eval()
    sample_idx = val_idx[:n]
    xb = torch.from_numpy(X[sample_idx]).to(device)
    with torch.no_grad():
        logits = model(xb)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)

    print("\n  Sample predictions (first {} val windows):".format(n))
    print(f"  {'idx':>6}  {'true':>6}  {'pred':>6}  {'prob':>8}  {'ok?':>4}")
    print("  " + "-" * 40)
    for i, si in enumerate(sample_idx):
        gt  = int(y[si])
        pr  = preds[i]
        pb  = probs[i]
        ok  = "OK" if gt == pr else "MISS"
        lbl = lambda v: "fake" if v == 1 else "real"
        print(f"  {si:>6}  {lbl(gt):>6}  {lbl(pr):>6}  {pb:>8.4f}  {ok:>4}")


def save_val_predictions(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    out_path: Path,
) -> None:
    model.eval()
    all_probs, all_preds = [], []
    batch = 256
    for start in range(0, len(val_idx), batch):
        idx = val_idx[start : start + batch]
        xb  = torch.from_numpy(X[idx]).to(device)
        with torch.no_grad():
            logits = model(xb)
            probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend((probs >= 0.5).astype(int).tolist())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "true_label", "pred_label", "prob_fake"])
        for i, si in enumerate(val_idx):
            w.writerow([si, int(y[si]), all_preds[i], f"{all_probs[i]:.6f}"])
    print(f"\n  Val predictions saved -> {out_path}")


# ===========================================================================
# 5.  Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="GRU baseline trainer — DeepShield liveness")
    parser.add_argument("--data",    type=Path, default=Path("data/sequences.npz"))
    parser.add_argument("--out",     type=Path, default=Path("models/gru_baseline.pt"))
    parser.add_argument("--epochs",  type=int,  default=20)
    parser.add_argument("--batch",   type=int,  default=32)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--hidden",  type=int,  default=128)
    parser.add_argument("--patience",type=int,  default=5,
                        help="Early-stop patience on val loss")
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--val-preds", type=Path, default=Path("data/val_predictions.csv"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading {args.data} ...")
    d = load_npz(args.data)

    X           = d["X"]                    # (N, 24, F) float32
    y           = d["y"]                    # (N,)       int8
    groups      = d["groups"]               # (N,)       int32
    session_ids = d["session_ids"]          # (S,)       str
    features    = list(d["features"])

    N, T, F = X.shape
    print(f"  Dataset : {N:,} samples  |  window={T}  |  features={F}  {features}")
    print(f"  Sessions: {len(session_ids)} unique  |  groups range [{groups.min()}-{groups.max()}]")
    print(f"  Labels  : real={int((y==0).sum()):,}  fake={int((y==1).sum()):,}")

    # ------------------------------------------------------------------
    # Session-grouped split (NO leakage)
    # ------------------------------------------------------------------
    train_idx, val_idx, train_sids, val_sids = group_split_indices(
        groups, session_ids, y, val_frac=0.20, seed=args.seed
    )

    # Leakage assertion — must pass before training starts
    assert len(set(train_sids) & set(val_sids)) == 0, \
        "LEAKAGE DETECTED: same session in both train and val splits!"

    print(f"\n  Train sessions ({len(train_sids)}): {[s[-8:] for s in train_sids]}")
    print(f"  Val   sessions ({len(val_sids)}):   {[s[-8:] for s in val_sids]}")
    print(f"  Train samples : {len(train_idx):,}  |  Val samples: {len(val_idx):,}")
    print(f"  Train real/fake: {int((y[train_idx]==0).sum())}/{int((y[train_idx]==1).sum())}")
    print(f"  Val   real/fake: {int((y[val_idx]==0).sum())}/{int((y[val_idx]==1).sum())}")

    train_loader, val_loader = make_loaders(X, y, train_idx, val_idx, args.batch)

    # ------------------------------------------------------------------
    # Model, loss, optimiser
    # ------------------------------------------------------------------
    model     = GRUClassifier(input_size=F, hidden_size=args.hidden).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {total_params:,}")

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss   = float("inf")
    patience_count  = 0
    best_epoch      = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>10}  "
          f"{'Acc':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
    print(f"{'='*68}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, _, _              = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   y_true, y_pred    = run_epoch(model, val_loader,   criterion, None,      device)

        m = compute_metrics(y_true, y_pred)
        elapsed = time.time() - t0

        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{m['acc']:>7.4f}  {m['f1']:>7.4f}  {m['precision']:>7.4f}  {m['recall']:>7.4f}"
              f"  [{elapsed:.1f}s]")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   val_loss,
                "features":   features,
                "hidden_size": args.hidden,
                "input_size": F,
            }, args.out)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  [Early stop] No improvement for {args.patience} epochs. "
                      f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
                break

    # ------------------------------------------------------------------
    # Final evaluation on best checkpoint
    # ------------------------------------------------------------------
    print(f"\n{'='*68}")
    print(f"  Loading best checkpoint (epoch {best_epoch}) for final evaluation ...")
    ckpt = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, y_true, y_pred = run_epoch(model, val_loader, criterion, None, device)
    m = compute_metrics(y_true, y_pred)

    print(f"\n  --- FINAL VALIDATION METRICS (epoch {best_epoch}) ---")
    print(f"  Accuracy  : {m['acc']:.4f}  ({m['acc']*100:.1f}%)")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Val Loss  : {best_val_loss:.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"               pred_real  pred_fake")
    cm = m["cm"]
    labels_ord = ["real (0)", "fake (1)"]
    for i, row_lbl in enumerate(labels_ord):
        print(f"  true_{row_lbl}  {cm[i, 0]:>9}  {cm[i, 1]:>9}")

    # ------------------------------------------------------------------
    # Sample predictions + CSV
    # ------------------------------------------------------------------
    show_sample_predictions(model, X, y, val_idx, device, n=10)
    save_val_predictions(model, X, y, val_idx, device, args.val_preds)

    print(f"\n  Best model saved -> {args.out}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
