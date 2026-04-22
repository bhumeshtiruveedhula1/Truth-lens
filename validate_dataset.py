import os
import pandas as pd
import numpy as np
from pathlib import Path

session_dir = Path("data/sessions")
sessions = sorted(session_dir.glob("session_*/frames.csv"))

ALL_SIGNAL_COLS = ["ear", "yaw", "pitch", "roll", "motion_score", "irregularity",
                   "temporal_score", "texture_score", "face_present"]

print(f"Found {len(sessions)} sessions\n")

all_dfs = []
per_session_summary = []

for s in sessions:
    df = pd.read_csv(s)
    sid = s.parent.name

    label_counts = df["label"].value_counts().to_dict()
    face_present_mask = df["face_present"] == 1
    face_df = df[face_present_mask]

    # --- A. Basic stats ---
    total = len(df)
    missing = df[ALL_SIGNAL_COLS].isnull().sum().to_dict()
    total_missing = sum(missing.values())

    # --- B. Signal sanity ---
    # Blink
    blink_rate = df["blink_detected"].mean()
    ear_std = df["ear"].std()
    ear_during_blink = df.loc[df["blink_detected"] == 1, "ear"].mean() if df["blink_detected"].sum() > 0 else np.nan
    ear_no_blink = df.loc[df["blink_detected"] == 0, "ear"].mean()

    # Head pose variance
    yaw_std = df["yaw"].std()
    pitch_std = df["pitch"].std()
    roll_std = df["roll"].std()

    # Motion
    motion_mean = df["motion_score"].mean()
    motion_std = df["motion_score"].std()

    # Flat signal detection (std < 1e-4 = suspicious)
    flat_signals = [col for col in ALL_SIGNAL_COLS if df[col].std() < 1e-4]

    # Zero-only signals
    zero_signals = [col for col in ALL_SIGNAL_COLS if (df[col] == 0).all()]

    # Temporal & texture
    temporal_mean = df["temporal_score"].mean()
    texture_mean = df["texture_score"].mean()

    label = df["label"].iloc[0] if df["label"].nunique() == 1 else "mixed"

    per_session_summary.append({
        "session_id": sid[-8:],
        "label": label,
        "total_frames": total,
        "face_present_pct": round(face_present_mask.mean() * 100, 1),
        "missing_values": total_missing,
        "blink_rate": round(blink_rate * 100, 2),
        "ear_std": round(ear_std, 4),
        "ear_blink_vs_no_blink": round(ear_during_blink - ear_no_blink, 4) if not np.isnan(ear_during_blink) else "N/A (no blinks)",
        "yaw_std": round(yaw_std, 4),
        "pitch_std": round(pitch_std, 4),
        "roll_std": round(roll_std, 4),
        "motion_mean": round(motion_mean, 4),
        "motion_std": round(motion_std, 4),
        "temporal_mean": round(temporal_mean, 5),
        "texture_mean": round(texture_mean, 4),
        "flat_signals": flat_signals,
        "zero_signals": zero_signals,
    })

    df["_session"] = sid
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)

# =============================================
# PRINT SECTION 1: Per-session summary
# =============================================
print("=" * 70)
print("SECTION 1: PER-SESSION SUMMARY")
print("=" * 70)
for ps in per_session_summary:
    print(f"\n[{ps['session_id']}] label={ps['label']}, frames={ps['total_frames']}, "
          f"face_ok={ps['face_present_pct']}%")
    print(f"  Missing values: {ps['missing_values']}")
    print(f"  Blink rate: {ps['blink_rate']}%  |  EAR std: {ps['ear_std']}")
    print(f"  EAR drop during blink: {ps['ear_blink_vs_no_blink']}")
    print(f"  Head pose std => yaw:{ps['yaw_std']}, pitch:{ps['pitch_std']}, roll:{ps['roll_std']}")
    print(f"  Motion => mean:{ps['motion_mean']}, std:{ps['motion_std']}")
    print(f"  Temporal mean: {ps['temporal_mean']}  |  Texture mean: {ps['texture_mean']}")
    if ps['flat_signals']:
        print(f"  !! FLAT SIGNALS: {ps['flat_signals']}")
    if ps['zero_signals']:
        print(f"  !! ALL-ZERO SIGNALS: {ps['zero_signals']}")

# =============================================
# SECTION 2: Real vs Fake signal comparison
# =============================================
print("\n")
print("=" * 70)
print("SECTION 2: REAL vs FAKE SIGNAL COMPARISON (across all sessions)")
print("=" * 70)

signal_cols = ["ear", "blink_detected", "yaw", "pitch", "roll",
               "motion_score", "irregularity", "temporal_score", "texture_score"]

real = combined[combined["label"] == "real"]
fake = combined[combined["label"] == "fake"]

print(f"\nTotal rows: {len(combined)} | Real: {len(real)} | Fake: {len(fake)}")
print(f"Class balance: Real={round(len(real)/len(combined)*100, 1)}%  Fake={round(len(fake)/len(combined)*100, 1)}%\n")

print(f"{'Signal':<20} {'Mean_Real':>12} {'Mean_Fake':>12} {'Std_Real':>12} {'Std_Fake':>12} {'Delta_Mean':>12} {'Separation':>12}")
print("-" * 95)

signal_verdicts = {}
for col in signal_cols:
    mr = real[col].mean()
    mf = fake[col].mean()
    sr = real[col].std()
    sf = fake[col].std()
    delta = abs(mr - mf)
    # Separation ratio: delta / average std (Cohen's d approximation)
    avg_std = (sr + sf) / 2 if (sr + sf) > 0 else 1e-9
    sep = delta / avg_std

    if sep > 0.5:
        verdict = "STRONG"
    elif sep > 0.1:
        verdict = "MODERATE"
    else:
        verdict = "WEAK"

    signal_verdicts[col] = {"delta": delta, "sep": sep, "verdict": verdict}

    print(f"{col:<20} {mr:>12.5f} {mf:>12.5f} {sr:>12.5f} {sf:>12.5f} {delta:>12.5f} {verdict:>12}")

# =============================================
# SECTION 3: Overall variability
# =============================================
print("\n")
print("=" * 70)
print("SECTION 3: SIGNAL VARIABILITY (std across ALL frames)")
print("=" * 70)
print(f"\n{'Signal':<20} {'Global_Std':>12} {'Min':>10} {'Max':>10} {'IsFlatGlobally':>16}")
print("-" * 72)
for col in signal_cols:
    g_std = combined[col].std()
    g_min = combined[col].min()
    g_max = combined[col].max()
    is_flat = "YES [FLAT]" if g_std < 1e-4 else "no"
    print(f"{col:<20} {g_std:>12.5f} {g_min:>10.4f} {g_max:>10.4f} {is_flat:>16}")

# =============================================
# SECTION 4: Problem Detection
# =============================================
print("\n")
print("=" * 70)
print("SECTION 4: PROBLEM DETECTION")
print("=" * 70)

problems = []

# Label imbalance
real_pct = len(real) / len(combined)
if real_pct < 0.3 or real_pct > 0.7:
    problems.append(f"LABEL IMBALANCE: Real={round(real_pct*100,1)}% vs Fake={round((1-real_pct)*100,1)}%")

# Sessions with same label only
labels_per_session = [(ps['session_id'], ps['label']) for ps in per_session_summary]
print(f"\nSession labels: {labels_per_session}")

# Check for near-zero or flat signals globally
for col in signal_cols:
    g_std = combined[col].std()
    if g_std < 1e-4:
        problems.append(f"FLAT SIGNAL GLOBALLY: {col} has near-zero std={g_std:.2e}")

# Check for signals always zero
for col in signal_cols:
    if (combined[col] == 0).mean() > 0.9:
        problems.append(f"MOSTLY ZERO: {col} is zero in {round((combined[col]==0).mean()*100,1)}% of frames")

# Blink detection check on fake sessions
for ps in per_session_summary:
    if ps["label"] == "fake" and ps["blink_rate"] > 5:
        problems.append(f"Session {ps['session_id']} (FAKE) has high blink rate: {ps['blink_rate']}%")

# Check weak signals
weak_sigs = [k for k, v in signal_verdicts.items() if v["verdict"] == "WEAK"]
strong_sigs = [k for k, v in signal_verdicts.items() if v["verdict"] == "STRONG"]

if len(strong_sigs) == 0:
    problems.append("NO STRONG SEPARATING SIGNALS FOUND - real vs fake are indistinguishable")

# Near-zero temporal_score in fake sessions
fake_temporal_mean = combined.loc[combined["label"] == "fake", "temporal_score"].mean()
real_temporal_mean = combined.loc[combined["label"] == "real", "temporal_score"].mean()
if fake_temporal_mean > real_temporal_mean * 0.5:
    problems.append(f"temporal_score NOT well separated: fake_mean={fake_temporal_mean:.5f}, real_mean={real_temporal_mean:.5f}")

# face_absent frames in fake sessions (face_present=0 with stale signal data)
stale_frames = combined[(combined["face_present"] == 0) & (combined["motion_score"] == 0)]
stale_pct = len(stale_frames) / len(combined) * 100
if stale_pct > 5:
    problems.append(f"STALE FROZEN FRAMES: {round(stale_pct,1)}% of all frames have face_present=0 & motion=0 (stale/carryover data)")

print("\nDetected Problems:")
if problems:
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}")
else:
    print("  None detected.")

# =============================================
# SECTION 5: FINAL VERDICT
# =============================================
print("\n")
print("=" * 70)
print("SECTION 5: FINAL VERDICT")
print("=" * 70)

print(f"\nStrong signals (useful for GRU): {strong_sigs}")
moderate_sigs = [k for k, v in signal_verdicts.items() if v["verdict"] == "MODERATE"]
print(f"Moderate signals: {moderate_sigs}")
print(f"Weak/useless signals: {weak_sigs}")

print("\n--- TOP ISSUES ---")
top3 = problems[:3]
for i, p in enumerate(top3, 1):
    print(f"  #{i}: {p}")

# Scoring: PASS or FAIL
fail_flags = 0
if len(strong_sigs) == 0: fail_flags += 2
if len(strong_sigs) < 2: fail_flags += 1
if real_pct < 0.2 or real_pct > 0.8: fail_flags += 2
stale_frame_count = len(combined[(combined["face_present"] == 0) & (combined["motion_score"] == 0)])
if stale_frame_count / len(combined) > 0.3: fail_flags += 2

if fail_flags >= 3:
    verdict_label = "FAIL"
    gru_ready = "NO - do NOT proceed to GRU training yet. Fix critical issues first."
elif fail_flags >= 1:
    verdict_label = "CONDITIONAL PASS"
    gru_ready = "MAYBE - dataset has usable signals but significant issues need addressing before GRU training."
else:
    verdict_label = "PASS"
    gru_ready = "YES - proceed to GRU training."

print(f"\n{'='*40}")
print(f"DATASET VERDICT:    {verdict_label}")
print(f"GRU TRAINING READY: {gru_ready}")
print(f"{'='*40}")

print(f"\nSession count: {len(sessions)}")
print(f"Total frames: {len(combined)}")
print(f"Real frames: {len(real)} | Fake frames: {len(fake)}")
print(f"Strong discriminating signals: {strong_sigs}")
