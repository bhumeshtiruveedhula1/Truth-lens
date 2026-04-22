"""
scripts/train_gru_v2.py -- Phase 3 GRU Retraining
Architecture UNCHANGED. Adds: weight_decay, best-epoch checkpoint.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

NPZ_PATH   = Path("data/sequences.npz")
OUT_PATH   = Path("models/gru_refined_v2.pt")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SEP = "-" * 64

EPOCHS=20; LR=5e-4; BATCH_SIZE=32; HIDDEN_SIZE=128; FOCAL_GAMMA=2.0
THRESHOLDS=[0.30, 0.35, 0.40, 0.45, 0.50]

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.where(targets==1, torch.sigmoid(logits).detach(), 1-torch.sigmoid(logits).detach())
        return ((1-pt)**self.gamma * bce).mean()

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
    rm   = y_true==0; fm = y_true==1
    racc = accuracy_score(y_true[rm], preds[rm]) if rm.any() else 0.0
    facc = accuracy_score(y_true[fm], preds[fm]) if fm.any() else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, real_acc=racc, fake_acc=facc)

# STEP 1
print(SEP); print("STEP 1: LOAD DATA"); print(SEP)
d = np.load(NPZ_PATH, allow_pickle=True)
X = d["X"].astype(np.float32); y = d["y"].astype(np.int64)
groups = d["groups"].astype(np.int32)
features = list(d["features"])
feat_mean = d["feat_mean"].astype(np.float32)
feat_std  = d["feat_std"].astype(np.float32)
assert X.shape[1]==24 and X.shape[2]==7
n_real=int((y==0).sum()); n_fake=int((y==1).sum())
print(f"  X.shape     : {X.shape}")
print(f"  Features    : {features}")
print(f"  Real        : {n_real:,} ({n_real/len(y)*100:.1f}%)")
print(f"  Fake        : {n_fake:,} ({n_fake/len(y)*100:.1f}%)")

# STEP 2
print(f"\n{SEP}"); print("STEP 2: TRAIN/VAL SPLIT (session-based)"); print(SEP)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=groups))
X_tr,y_tr = X[train_idx],y[train_idx]
X_va,y_va = X[val_idx],  y[val_idx]
print(f"  Train : {len(X_tr):,}  real={int((y_tr==0).sum()):,}  fake={int((y_tr==1).sum()):,}")
print(f"  Val   : {len(X_va):,}  real={int((y_va==0).sum()):,}  fake={int((y_va==1).sum()):,}")
overlap = set(groups[train_idx].tolist()) & set(groups[val_idx].tolist())
assert len(overlap)==0, f"LEAKAGE: {overlap}"
print(f"  Session leakage: 0  OK")

# STEP 3+4
print(f"\n{SEP}"); print("STEP 3+4: MODEL + CONFIG"); print(SEP)
device = torch.device("cpu")
model  = GRUModel(7, HIDDEN_SIZE).to(device)
loss_fn   = FocalLoss(FOCAL_GAMMA)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
print(f"  GRU(7->128)->Linear(128->1)  FocalLoss(g={FOCAL_GAMMA})  Adam(lr={LR} wd=1e-4)")
print(f"  Epochs={EPOCHS}  Batch={BATCH_SIZE}  Best-epoch checkpoint enabled")
tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)))
va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.float32)))
tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

# STEP 5
print(f"\n{SEP}"); print("STEP 5: TRAINING (* = new best val_loss)"); print(SEP)
print(f"  {'Ep':>3}  {'TrLoss':>8}  {'VaLoss':>8}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
print(f"  {'-'*60}")

history=[]; best_val=float("inf"); best_epoch=0; best_state=None

for epoch in range(1, EPOCHS+1):
    model.train(); tr_sum=0.0
    for xb,yb in tr_dl:
        xb,yb = xb.to(device),yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); optimizer.step()
        tr_sum += loss.item()*len(xb)
    tr_loss = tr_sum/len(tr_ds)

    model.eval(); va_sum=0.0; probs_all=[]; true_all=[]
    with torch.no_grad():
        for xb,yb in va_dl:
            xb,yb = xb.to(device),yb.to(device)
            lgts  = model(xb)
            va_sum += loss_fn(lgts, yb).item()*len(xb)
            probs_all.extend(torch.sigmoid(lgts).cpu().numpy().tolist())
            true_all.extend(yb.cpu().numpy().tolist())
    va_loss  = va_sum/len(va_ds)
    probs_np = np.array(probs_all); true_np = np.array(true_all,dtype=int)
    m = metrics_at(true_np, probs_np, 0.50)

    mark = ""
    if va_loss < best_val:
        best_val=va_loss; best_epoch=epoch
        best_state={k:v.clone() for k,v in model.state_dict().items()}
        mark=" *"

    print(f"  {epoch:>3}  {tr_loss:>8.4f}  {va_loss:>8.4f}  "
          f"{m['acc']:>7.4f}  {m['prec']:>7.4f}  {m['rec']:>7.4f}  {m['f1']:>7.4f}{mark}")
    history.append(dict(epoch=epoch,tr_loss=tr_loss,va_loss=va_loss,**m))

print(f"\n  Best epoch : {best_epoch}  (val_loss={best_val:.4f})")
print(f"  Restoring best weights...")
model.load_state_dict(best_state)

# Re-eval with best weights
model.eval(); probs_all=[]; true_all=[]
with torch.no_grad():
    for xb,yb in va_dl:
        xb,yb = xb.to(device),yb.to(device)
        probs_all.extend(torch.sigmoid(model(xb)).cpu().numpy().tolist())
        true_all.extend(yb.cpu().numpy().tolist())
probs_np = np.array(probs_all); true_np = np.array(true_all,dtype=int)

best_h = history[best_epoch-1]
print(f"  Best-epoch val @ thr=0.50: acc={best_h['acc']:.4f}  f1={best_h['f1']:.4f}")
if best_h["tr_loss"]<0.05 and best_h["va_loss"]>0.25:
    print("  WARNING: overfitting detected even at best epoch")
else:
    print("  Overfitting: not detected at best epoch")

# STEP 6
print(f"\n{SEP}"); print("STEP 6: THRESHOLD SWEEP"); print(SEP)
print(f"  {'Thr':>6}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'RealAcc':>9}  {'FakeAcc':>9}")
print(f"  {'-'*68}")
sweep=[]
for thr in THRESHOLDS:
    m = metrics_at(true_np, probs_np, thr)
    sweep.append((thr,m))
    print(f"  {thr:>6.2f}  {m['acc']:>7.4f}  {m['prec']:>7.4f}  {m['rec']:>7.4f}  "
          f"{m['f1']:>7.4f}  {m['real_acc']:>9.4f}  {m['fake_acc']:>9.4f}")

# STEP 7
print(f"\n{SEP}"); print("STEP 7: THRESHOLD SELECTION"); print(SEP)
print("  Rule: real_acc >= 0.80 first, then max F1")
best_thr=0.50; best_f1=-1.0; best_m=None
for thr,m in sweep:
    if m["real_acc"]>=0.80 and m["f1"]>best_f1:
        best_f1=m["f1"]; best_thr=thr; best_m=m
if best_m is None:
    print("  NOTE: no thr achieved real_acc >= 0.80 -- fallback to best F1")
    for thr,m in sweep:
        if m["f1"]>best_f1: best_f1=m["f1"]; best_thr=thr; best_m=m
print(f"  Best thr={best_thr}  acc={best_m['acc']:.4f}  prec={best_m['prec']:.4f}  "
      f"rec={best_m['rec']:.4f}  f1={best_m['f1']:.4f}  "
      f"real_acc={best_m['real_acc']:.4f}  fake_acc={best_m['fake_acc']:.4f}")

# STEP 8
print(f"\n{SEP}"); print("STEP 8: SAVE"); print(SEP)
ckpt = dict(model_state=model.state_dict(), features=features,
            best_threshold=best_thr, input_size=7, hidden_size=HIDDEN_SIZE,
            feat_mean=feat_mean, feat_std=feat_std)
torch.save(ckpt, OUT_PATH)
ck2 = torch.load(OUT_PATH, map_location="cpu", weights_only=False)
assert ck2["features"]==features and ck2["input_size"]==7
print(f"  Saved: {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.1f} KB)  Reload: PASS")
print(f"  best_threshold={best_thr}  features={features}")

# STEP 9
print(f"\n{SEP}"); print("STEP 9: FINAL REPORT"); print(SEP)
print(f"  Dataset       : {X.shape[0]:,} seq  train={len(X_tr):,}  val={len(X_va):,}")
print(f"  Best epoch    : {best_epoch}/{EPOCHS}  val_loss={best_val:.4f}")
print(f"  Val acc       : {best_h['acc']:.4f}")
print(f"  Val precision : {best_h['prec']:.4f}")
print(f"  Val recall    : {best_h['rec']:.4f}")
print(f"  Val F1        : {best_h['f1']:.4f}")
print(f"  Best thr      : {best_thr}")
print(f"  Real acc @thr : {best_m['real_acc']:.4f}")
print(f"  Fake acc @thr : {best_m['fake_acc']:.4f}")
print(f"  Overfitting   : {'YES' if best_h['tr_loss']<0.05 and best_h['va_loss']>0.25 else 'NO'}")
print(f"  Saved to      : {OUT_PATH}")
print(f"  Next step     : update inference.py model_path to models/gru_refined_v2.pt")
print(SEP)
