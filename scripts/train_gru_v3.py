"""
scripts/train_gru_v3.py -- Phase 3b: Fixed split + retrain
Ensures validation set has ~50/50 real/fake sessions before training.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

NPZ_PATH = Path("data/sequences.npz")
OUT_PATH = Path("models/gru_refined_v3.pt")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SEP = "-" * 64

EPOCHS = 15; LR = 5e-4; BATCH_SIZE = 32; HIDDEN_SIZE = 128; FOCAL_GAMMA = 2.0
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]
REAL_ACC_MIN = 0.75
FAKE_ACC_MIN = 0.70


# ── Model + Loss (UNCHANGED) ──────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.where(targets == 1,
                          torch.sigmoid(logits).detach(),
                          1 - torch.sigmoid(logits).detach())
        return ((1 - pt) ** self.gamma * bce).mean()

class GRUModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=128):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h.squeeze(0)).squeeze(-1)

def metrics_at(y_true, probs, thr):
    preds = (probs >= thr).astype(int)
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true, preds, zero_division=0)
    rm = y_true == 0; fm = y_true == 1
    racc = accuracy_score(y_true[rm], preds[rm]) if rm.any() else 0.0
    facc = accuracy_score(y_true[fm], preds[fm]) if fm.any() else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, real_acc=racc, fake_acc=facc)


# ── STEP 1: Load ──────────────────────────────────────────────────────────────
print(SEP); print("STEP 1: LOAD DATA"); print(SEP)

d = np.load(NPZ_PATH, allow_pickle=True)
X        = d["X"].astype(np.float32)
y        = d["y"].astype(np.int64)
groups   = d["groups"].astype(np.int32)
features = list(d["features"])
feat_mean = d["feat_mean"].astype(np.float32)
feat_std  = d["feat_std"].astype(np.float32)
session_ids = list(d["session_ids"]) if "session_ids" in d else []

assert X.shape[1] == 24 and X.shape[2] == 7
n_real = int((y == 0).sum()); n_fake = int((y == 1).sum())
print(f"  X.shape   : {X.shape}")
print(f"  Real seqs : {n_real:,}  Fake seqs : {n_fake:,}")
print(f"  Sessions  : {len(np.unique(groups))}")


# ── STEP 2: Build balanced session-level split ────────────────────────────────
print(f"\n{SEP}"); print("STEP 2: BALANCED SESSION SPLIT"); print(SEP)

# Determine per-session label (majority vote)
unique_sessions = np.unique(groups)
session_labels  = {}   # session_idx -> 0 (real) or 1 (fake)
for sid in unique_sessions:
    mask = groups == sid
    lbl  = int(np.round(y[mask].mean()))   # majority label
    session_labels[sid] = lbl

real_sessions = sorted([s for s, l in session_labels.items() if l == 0])
fake_sessions = sorted([s for s, l in session_labels.items() if l == 1])
print(f"  Real sessions : {len(real_sessions)}")
print(f"  Fake sessions : {len(fake_sessions)}")

# Target val: 2 real + 2 fake sessions (to keep val balanced regardless of total count)
# Use OPTION B: try multiple seeds and pick best-balanced split
SEEDS = [0, 7, 13, 21, 42, 99]
best_seed   = None
best_balance = float("inf")
best_split   = None

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    r_val = rng.choice(real_sessions, size=max(2, len(real_sessions)//5), replace=False).tolist()
    f_val = rng.choice(fake_sessions, size=max(2, len(fake_sessions)//5), replace=False).tolist()
    val_sids  = set(r_val + f_val)
    tr_sids   = set(unique_sessions) - val_sids

    val_mask  = np.isin(groups, list(val_sids))
    tr_mask   = np.isin(groups, list(tr_sids))
    y_va_tmp  = y[val_mask]; y_tr_tmp = y[tr_mask]

    va_real = int((y_va_tmp == 0).sum()); va_fake = int((y_va_tmp == 1).sum())
    va_total = va_real + va_fake
    if va_total == 0:
        continue
    va_ratio  = va_real / va_total
    imbalance = abs(va_ratio - 0.50)

    tr_real = int((y_tr_tmp == 0).sum()); tr_fake = int((y_tr_tmp == 1).sum())
    print(f"  seed={seed:>3}  val real={va_real:>4} fake={va_fake:>4} "
          f"({va_ratio*100:.0f}% real)  "
          f"tr real={tr_real} fake={tr_fake}  "
          f"|balance|={imbalance:.3f}")

    if imbalance < best_balance:
        best_balance = imbalance
        best_seed    = seed
        best_split   = (list(tr_sids), list(val_sids), va_ratio)

tr_sids, val_sids, chosen_ratio = best_split
print(f"\n  CHOSEN seed={best_seed}  val real/fake ratio={chosen_ratio*100:.1f}%/{ (1-chosen_ratio)*100:.1f}%")

tr_mask  = np.isin(groups, tr_sids)
val_mask = np.isin(groups, list(val_sids))
X_tr, y_tr = X[tr_mask],  y[tr_mask]
X_va, y_va = X[val_mask], y[val_mask]

tr_real = int((y_tr == 0).sum()); tr_fake = int((y_tr == 1).sum())
va_real = int((y_va == 0).sum()); va_fake = int((y_va == 1).sum())
print(f"\n  Train : {len(X_tr):,}  real={tr_real:,} ({tr_real/len(X_tr)*100:.0f}%)  fake={tr_fake:,} ({tr_fake/len(X_tr)*100:.0f}%)")
print(f"  Val   : {len(X_va):,}  real={va_real:,} ({va_real/len(X_va)*100:.0f}%)  fake={va_fake:,} ({va_fake/len(X_va)*100:.0f}%)")

# Leakage check
overlap = set(tr_sids) & set(val_sids)
assert len(overlap) == 0, f"LEAKAGE: {overlap}"
print(f"  Session leakage: 0  OK")


# ── STEP 3+4: Train with best-epoch checkpoint ────────────────────────────────
print(f"\n{SEP}"); print("STEP 3+4: TRAIN (best-epoch checkpoint)"); print(SEP)
print(f"  GRU(7->128)->Linear  FocalLoss(g={FOCAL_GAMMA})  Adam(lr={LR} wd=1e-4)")
print(f"  Epochs={EPOCHS}  Batch={BATCH_SIZE}")

device = torch.device("cpu")
model  = GRUModel(7, HIDDEN_SIZE).to(device)
loss_fn   = FocalLoss(FOCAL_GAMMA)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)))
va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.float32)))
tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n  {'Ep':>3}  {'TrLoss':>8}  {'VaLoss':>8}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'RealA':>7}  {'FakeA':>7}")
print(f"  {'-'*76}")

history=[]; best_val=float("inf"); best_epoch=0; best_state=None

for epoch in range(1, EPOCHS + 1):
    model.train(); tr_sum = 0.0
    for xb, yb in tr_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); optimizer.step()
        tr_sum += loss.item() * len(xb)
    tr_loss = tr_sum / len(tr_ds)

    model.eval(); va_sum = 0.0; probs_all = []; true_all = []
    with torch.no_grad():
        for xb, yb in va_dl:
            xb, yb = xb.to(device), yb.to(device)
            lgts = model(xb)
            va_sum += loss_fn(lgts, yb).item() * len(xb)
            probs_all.extend(torch.sigmoid(lgts).cpu().numpy().tolist())
            true_all.extend(yb.cpu().numpy().tolist())
    va_loss  = va_sum / len(va_ds)
    probs_np = np.array(probs_all); true_np = np.array(true_all, dtype=int)
    m = metrics_at(true_np, probs_np, 0.50)

    mark = ""
    if va_loss < best_val:
        best_val = va_loss; best_epoch = epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        mark = " *"

    print(f"  {epoch:>3}  {tr_loss:>8.4f}  {va_loss:>8.4f}  "
          f"{m['acc']:>7.4f}  {m['prec']:>7.4f}  {m['rec']:>7.4f}  {m['f1']:>7.4f}  "
          f"{m['real_acc']:>7.4f}  {m['fake_acc']:>7.4f}{mark}")
    history.append(dict(epoch=epoch, tr_loss=tr_loss, va_loss=va_loss, **m))

print(f"\n  Best epoch : {best_epoch}/{EPOCHS}  val_loss={best_val:.4f}")
model.load_state_dict(best_state)

# Re-eval with best weights
model.eval(); probs_all = []; true_all = []
with torch.no_grad():
    for xb, yb in va_dl:
        xb, yb = xb.to(device), yb.to(device)
        probs_all.extend(torch.sigmoid(model(xb)).cpu().numpy().tolist())
        true_all.extend(yb.cpu().numpy().tolist())
probs_np = np.array(probs_all); true_np = np.array(true_all, dtype=int)
best_h = history[best_epoch - 1]


# ── STEP 5: Threshold sweep ───────────────────────────────────────────────────
print(f"\n{SEP}"); print("STEP 5: THRESHOLD SWEEP"); print(SEP)
print(f"  {'Thr':>6}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'RealAcc':>9}  {'FakeAcc':>9}  Status")
print(f"  {'-'*80}")

sweep = []
for thr in THRESHOLDS:
    m = metrics_at(true_np, probs_np, thr)
    meets = m["real_acc"] >= REAL_ACC_MIN and m["fake_acc"] >= FAKE_ACC_MIN
    status = "PASS" if meets else f"FAIL(r={m['real_acc']:.2f}<{REAL_ACC_MIN} f={m['fake_acc']:.2f}<{FAKE_ACC_MIN})"
    sweep.append((thr, m, meets))
    print(f"  {thr:>6.2f}  {m['acc']:>7.4f}  {m['prec']:>7.4f}  {m['rec']:>7.4f}  "
          f"{m['f1']:>7.4f}  {m['real_acc']:>9.4f}  {m['fake_acc']:>9.4f}  {status}")


# ── STEP 6: Select threshold by success criteria ──────────────────────────────
print(f"\n{SEP}"); print("STEP 6: SELECT THRESHOLD"); print(SEP)
print(f"  Criteria: real_acc >= {REAL_ACC_MIN}  AND  fake_acc >= {FAKE_ACC_MIN}")

# Priority: thresholds that PASS both criteria, pick best F1
passing = [(thr, m) for thr, m, ok in sweep if ok]
if passing:
    best_thr, best_m = max(passing, key=lambda x: x[1]["f1"])
    print(f"  {len(passing)} threshold(s) passed criteria.")
    print(f"  Selected threshold : {best_thr}  (best F1={best_m['f1']:.4f})")
else:
    # Fallback: find threshold with best harmonic mean of real_acc and fake_acc
    print("  WARNING: No threshold met both criteria.")
    print("  Fallback: selecting threshold with best harmonic mean(real_acc, fake_acc)")
    best_thr, best_m = None, None
    best_hm = -1.0
    for thr, m, _ in sweep:
        r, f = m["real_acc"], m["fake_acc"]
        hm = 2 * r * f / (r + f) if (r + f) > 0 else 0.0
        if hm > best_hm:
            best_hm = hm; best_thr = thr; best_m = m
    print(f"  Selected threshold : {best_thr}  (harmonic_mean={best_hm:.4f})")

print(f"\n  Final metrics @ thr={best_thr}:")
print(f"    Accuracy  : {best_m['acc']:.4f}")
print(f"    Precision : {best_m['prec']:.4f}")
print(f"    Recall    : {best_m['rec']:.4f}")
print(f"    F1        : {best_m['f1']:.4f}")
print(f"    Real Acc  : {best_m['real_acc']:.4f}  (target >= {REAL_ACC_MIN})")
print(f"    Fake Acc  : {best_m['fake_acc']:.4f}  (target >= {FAKE_ACC_MIN})")
criteria_met = best_m["real_acc"] >= REAL_ACC_MIN and best_m["fake_acc"] >= FAKE_ACC_MIN
print(f"    Criteria  : {'PASS - proceed to deployment' if criteria_met else 'FAIL - more data or tuning needed'}")


# ── STEP 7: Save ──────────────────────────────────────────────────────────────
print(f"\n{SEP}"); print("STEP 7: SAVE"); print(SEP)

ckpt = dict(
    model_state=model.state_dict(),
    features=features,
    best_threshold=best_thr,
    input_size=7,
    hidden_size=HIDDEN_SIZE,
    feat_mean=feat_mean,
    feat_std=feat_std,
)
torch.save(ckpt, OUT_PATH)
ck2 = torch.load(OUT_PATH, map_location="cpu", weights_only=False)
assert ck2["features"] == features and ck2["input_size"] == 7
print(f"  Saved : {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
print(f"  Reload: PASS")


# ── STEP 8: Final report ──────────────────────────────────────────────────────
print(f"\n{SEP}"); print("STEP 8: FINAL REPORT"); print(SEP)
print(f"  Split (seed={best_seed})")
print(f"    Train : {len(X_tr):,}  real={tr_real:,}  fake={tr_fake:,}")
print(f"    Val   : {len(X_va):,}  real={va_real:,}  fake={va_fake:,}")
print(f"  Training")
print(f"    Epochs run    : {EPOCHS}")
print(f"    Best epoch    : {best_epoch}  val_loss={best_val:.4f}")
print(f"    Best val acc  : {best_h['acc']:.4f}")
print(f"    Best val F1   : {best_h['f1']:.4f}")
print(f"    Best real_acc : {best_h['real_acc']:.4f}")
print(f"    Best fake_acc : {best_h['fake_acc']:.4f}")
print(f"  Threshold sweep")
print(f"    Best thr      : {best_thr}")
print(f"    Real Acc      : {best_m['real_acc']:.4f}  ({'OK' if best_m['real_acc'] >= REAL_ACC_MIN else 'BELOW TARGET'})")
print(f"    Fake Acc      : {best_m['fake_acc']:.4f}  ({'OK' if best_m['fake_acc'] >= FAKE_ACC_MIN else 'BELOW TARGET'})")
print(f"  Criteria met    : {'YES' if criteria_met else 'NO'}")
print(f"  Model saved     : {OUT_PATH}")
print(f"  Next step       : wire models/gru_refined_v3.pt into inference.py")
print(SEP)
