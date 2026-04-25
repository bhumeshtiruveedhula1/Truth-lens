"""
scripts/test_deepfake_model.py
=================================
Standalone validation script for deepfake_efficientnet_final.pt.
Evaluates THREE test categories independently:
  - REAL  : val/real  (held-out webcam + celeb_real)
  - FAKE  : val/fake  (celeb_df frames never seen in training)
  - OBS   : obs_p3/p4 original captures (val identity group, NOT in train)

PASS criteria (aligned with multi-layer system context):
  REAL  avg < 0.40  (low false-positive rate)
  FAKE  avg > 0.70  (high deepfake detection)
  OBS   avg > 0.60  (OBS injection detection — CRITICAL)

Usage:
  python scripts/test_deepfake_model.py
  python scripts/test_deepfake_model.py --live        # webcam test
  python scripts/test_deepfake_model.py --n 150       # frames per category
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "models/deepfake_efficientnet_final.pt"

# Test sources — use held-out data only
VAL_REAL    = ROOT / "data/deepfake_final/val/real"
VAL_FAKE    = ROOT / "data/deepfake_final/val/fake"

# OBS: use val identity group (obs_p3, obs_p4) — NEVER seen during training
OBS_DIRS    = [
    ROOT / "data/deepfake_dataset/fake/obs_p3_v1",
    ROOT / "data/deepfake_dataset/fake/obs_p3_v2",
    ROOT / "data/deepfake_dataset/fake/obs_p3_v3",
    ROOT / "data/deepfake_dataset/fake/obs_p4_v1",
    ROOT / "data/deepfake_dataset/fake/obs_p4_v2",
    ROOT / "data/deepfake_dataset/fake/obs_p4_v3",
]

IMG_SIZE  = 224
IMG_EXTS  = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SEP = "=" * 60


# ── Device ────────────────────────────────────────────────────────────────────

def setup_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (GPU not available)")
    return device


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> nn.Module:
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    model = efficientnet_b0(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, 1)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.eval()
    model.to(device)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"[Model] Loaded {MODEL_PATH.name}  ({size_mb:.1f}MB)  device={device}")
    return model


# ── Preprocessing — EXACT SAME as training ────────────────────────────────────

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess(face_bgr: np.ndarray) -> torch.Tensor | None:
    """BGR numpy -> normalised tensor.  Returns None if input is invalid."""
    if face_bgr is None or face_bgr.size == 0:
        return None
    if face_bgr.mean() < 5:
        return None      # black / blank frame
    try:
        rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        return _transform(pil)
    except Exception:
        return None


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(face_bgr: np.ndarray, model: nn.Module,
            device: torch.device) -> float | None:
    """
    Returns P(fake) in [0, 1], or None if frame is invalid.
    """
    t = preprocess(face_bgr)
    if t is None:
        return None
    batch  = t.unsqueeze(0).to(device, non_blocking=True)
    logit  = model(batch)
    prob   = torch.sigmoid(logit).item()
    return float(prob)


# ── Static file evaluation ────────────────────────────────────────────────────

def eval_directory(
    paths: list[Path],
    model: nn.Module,
    device: torch.device,
    n: int,
    label: str,
) -> dict:
    """Run model on n randomly sampled images from paths. Return stats dict."""
    random.shuffle(paths)
    sample = paths[:n]
    probs  : list[float] = []
    skipped = 0

    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            skipped += 1
            continue
        # Ensure 224x224 input (should already be from dataset pipeline)
        if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        score = predict(img, model, device)
        if score is None:
            skipped += 1
            continue
        probs.append(score)

    if not probs:
        return {"label": label, "n": 0, "skipped": skipped, "error": "no valid frames"}

    arr = np.array(probs)
    return {
        "label"     : label,
        "n"         : len(arr),
        "skipped"   : skipped,
        "avg"       : float(arr.mean()),
        "min"       : float(arr.min()),
        "max"       : float(arr.max()),
        "std"       : float(arr.std()),
        "over_07"   : int((arr > 0.7).sum()),
        "under_03"  : int((arr < 0.3).sum()),
        "probs"     : probs,
    }


# ── Live webcam evaluation ────────────────────────────────────────────────────

def eval_live(model: nn.Module, device: torch.device, n: int) -> dict:
    """Capture n frames from webcam and run inference."""
    try:
        import mediapipe as mp
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.35
        )
        use_mp = True
    except Exception:
        use_mp = False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"label": "REAL_LIVE", "n": 0, "error": "camera not available"}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(20):
        cap.read()   # warm-up

    probs: list[float] = []
    skipped = 0

    print(f"  Capturing {n} frames from webcam (press q to stop early)...")
    while len(probs) < n:
        ret, frame = cap.read()
        if not ret or (frame is not None and frame.mean() < 5):
            skipped += 1
            continue

        # Face crop
        face = None
        if use_mp:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.process(rgb)
            if res and res.detections:
                b  = res.detections[0].location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x1 = max(0, int(b.xmin * w))
                y1 = max(0, int(b.ymin * h))
                x2 = min(w, int((b.xmin + b.width)  * w))
                y2 = min(h, int((b.ymin + b.height) * h))
                if (x2 - x1) > 40 and (y2 - y1) > 40:
                    face = frame[y1:y2, x1:x2]
        if face is None:
            face = frame   # fallback: use full frame

        score = predict(face, model, device)
        if score is not None:
            probs.append(score)
        else:
            skipped += 1

        # Live overlay
        status = f"Captured {len(probs)}/{n}  score={score:.3f}" if score else f"Captured {len(probs)}/{n}"
        display = frame.copy()
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        cv2.imshow("Live Validation", display)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    if use_mp:
        detector.close()

    if not probs:
        return {"label": "REAL_LIVE", "n": 0, "skipped": skipped, "error": "no valid frames"}

    arr = np.array(probs)
    return {
        "label"  : "REAL_LIVE",
        "n"      : len(arr),
        "skipped": skipped,
        "avg"    : float(arr.mean()),
        "min"    : float(arr.min()),
        "max"    : float(arr.max()),
        "std"    : float(arr.std()),
        "over_07": int((arr > 0.7).sum()),
        "under_03": int((arr < 0.3).sum()),
        "probs"  : probs,
    }


# ── Temporal consistency analysis ────────────────────────────────────────────

def temporal_analysis(probs: list[float], label: str) -> str:
    """Return brief stability description."""
    if len(probs) < 3:
        return "insufficient frames"
    arr = np.array(probs)
    # Count spikes: frames more than 2 std devs from mean
    spikes = int(np.sum(np.abs(arr - arr.mean()) > 2 * arr.std()))
    # Consecutive runs (rough temporal consistency)
    diffs = np.abs(np.diff(arr))
    avg_frame_delta = float(diffs.mean())

    stability = "STABLE" if arr.std() < 0.15 else ("MODERATE" if arr.std() < 0.25 else "NOISY")
    return (f"stability={stability}  spikes={spikes}/{len(probs)}"
            f"  avg_frame_delta={avg_frame_delta:.3f}")


# ── Pass/fail judgment ────────────────────────────────────────────────────────

PASS_CRITERIA = {
    "REAL"      : lambda r: r["avg"] < 0.40 and r["std"] < 0.25,
    "REAL_LIVE" : lambda r: r["avg"] < 0.40 and r["std"] < 0.25,
    "FAKE"      : lambda r: r["avg"] > 0.70,
    "OBS"       : lambda r: r["avg"] > 0.60,
}

FAIL_REASONS = {
    "REAL"      : "avg P(fake|real) > 0.40 -- domain mismatch risk",
    "REAL_LIVE" : "avg P(fake|real_live) > 0.40 -- false positive on webcam",
    "FAKE"      : "avg P(fake|fake) < 0.70 -- model missing deepfakes",
    "OBS"       : "avg P(fake|obs) < 0.60 -- system VULNERABLE to OBS injection",
}


def verdict(result: dict) -> tuple[bool, str]:
    lbl   = result.get("label", "")
    check = PASS_CRITERIA.get(lbl)
    if "error" in result or result.get("n", 0) == 0:
        return False, "no valid frames evaluated"
    if check is None:
        return True, "no criteria defined"
    ok = check(result)
    reason = "" if ok else FAIL_REASONS.get(lbl, "check failed")
    return ok, reason


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> bool:
    print("\n" + SEP)
    print("VALIDATION REPORT")
    print(SEP)

    # Table header
    print(f"\n{'Category':<14} {'N':>5} {'Avg':>7} {'Min':>7} {'Max':>7} "
          f"{'Std':>7} {'>0.7':>6} {'<0.3':>6}  Status")
    print("-" * 75)

    all_pass = True
    fail_reasons = []

    for r in results:
        if "error" in r:
            print(f"  {r['label']:<12}  ERROR: {r['error']}")
            all_pass = False
            continue

        ok, reason = verdict(r)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
            fail_reasons.append(f"  [{r['label']}] {reason}")

        print(f"  {r['label']:<12} {r['n']:>5} {r['avg']:>7.3f} {r['min']:>7.3f}"
              f" {r['max']:>7.3f} {r['std']:>7.3f} {r['over_07']:>6} {r['under_03']:>6}"
              f"  {status}")

    # Temporal analysis
    print("\nTemporal Consistency:")
    for r in results:
        if "probs" in r and r.get("n", 0) > 0:
            ta = temporal_analysis(r["probs"], r["label"])
            print(f"  {r['label']:<14}: {ta}")

    # False positive / false negative
    print("\nError Rates:")
    real_results = [r for r in results if r.get("label", "").startswith("REAL") and "probs" in r]
    fake_results = [r for r in results if r.get("label", "") in ("FAKE", "OBS") and "probs" in r]

    if real_results:
        all_real = []
        for r in real_results:
            all_real.extend(r["probs"])
        fp_rate = sum(1 for p in all_real if p >= 0.5) / len(all_real)
        print(f"  False Positive Rate (real classified as fake) : {fp_rate:.3f}  "
              f"({int(fp_rate*len(all_real))}/{len(all_real)})")

    if fake_results:
        all_fake = []
        for r in fake_results:
            all_fake.extend(r["probs"])
        fn_rate = sum(1 for p in all_fake if p < 0.5) / len(all_fake)
        print(f"  False Negative Rate (fake classified as real) : {fn_rate:.3f}  "
              f"({int(fn_rate*len(all_fake))}/{len(all_fake)})")

    # Final verdict
    print("\n" + SEP)
    if all_pass:
        print("  FINAL VERDICT: PASS")
        print("  Model cleared for integration into deepfake inference engine.")
        print("  Next: update agent/ml/deepfake_inference.py to load")
        print(f"        {MODEL_PATH.name}")
    else:
        print("  FINAL VERDICT: FAIL")
        print("  Reasons:")
        for r in fail_reasons:
            print(r)
        print("  DO NOT integrate until issues are resolved.")
    print(SEP)

    return all_pass


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Standalone deepfake model validation")
    p.add_argument("--n",     type=int, default=120,
                   help="Frames to evaluate per category (default 120)")
    p.add_argument("--live",  action="store_true",
                   help="Add live webcam test in addition to static files")
    p.add_argument("--seed",  type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    random.seed(args.seed)

    device = setup_device()
    model  = load_model(device)

    n = args.n
    results: list[dict] = []

    # ── Test 1: REAL (val/real — never in training) ───────────────────────────
    print(f"\n{SEP}")
    print("TEST 1 — REAL IMAGES (val/real, held-out)")
    print(SEP)
    real_paths = sorted(p for p in VAL_REAL.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not real_paths:
        print(f"  [WARN] No images found in {VAL_REAL}")
    else:
        print(f"  Source: {VAL_REAL}  ({len(real_paths)} images available)")
        t0 = time.time()
        r  = eval_directory(real_paths, model, device, n, "REAL")
        print(f"  Evaluated {r.get('n', 0)} frames in {time.time()-t0:.1f}s"
              f"  (skipped {r.get('skipped', 0)})")
        if "avg" in r:
            print(f"  avg={r['avg']:.4f}  std={r['std']:.4f}  "
                  f"min={r['min']:.4f}  max={r['max']:.4f}")
        results.append(r)

    # ── Test 2: FAKE — Celeb-DF (val/fake) ───────────────────────────────────
    print(f"\n{SEP}")
    print("TEST 2 — FAKE / CELEB-DF (val/fake, held-out)")
    print(SEP)
    fake_paths = sorted(p for p in VAL_FAKE.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not fake_paths:
        print(f"  [WARN] No images found in {VAL_FAKE}")
    else:
        print(f"  Source: {VAL_FAKE}  ({len(fake_paths)} images available)")
        t0 = time.time()
        r  = eval_directory(fake_paths, model, device, n, "FAKE")
        print(f"  Evaluated {r.get('n', 0)} frames in {time.time()-t0:.1f}s"
              f"  (skipped {r.get('skipped', 0)})")
        if "avg" in r:
            print(f"  avg={r['avg']:.4f}  std={r['std']:.4f}  "
                  f"min={r['min']:.4f}  max={r['max']:.4f}")
        results.append(r)

    # ── Test 3: OBS — val identity group (never in training) ─────────────────
    print(f"\n{SEP}")
    print("TEST 3 — OBS INJECTION (obs_p3/p4 — val identity, NOT in train)")
    print(SEP)
    obs_paths = []
    for d in OBS_DIRS:
        if d.exists():
            obs_paths.extend(p for p in d.glob("*.jpg"))
        else:
            print(f"  [WARN] Missing: {d.name}")
    if not obs_paths:
        print("  [SKIP] No OBS test frames found.")
    else:
        print(f"  Sources: {len([d for d in OBS_DIRS if d.exists()])} OBS sessions"
              f"  ({len(obs_paths)} images available)")
        t0 = time.time()
        r  = eval_directory(obs_paths, model, device, n, "OBS")
        print(f"  Evaluated {r.get('n', 0)} frames in {time.time()-t0:.1f}s"
              f"  (skipped {r.get('skipped', 0)})")
        if "avg" in r:
            print(f"  avg={r['avg']:.4f}  std={r['std']:.4f}  "
                  f"min={r['min']:.4f}  max={r['max']:.4f}")
        results.append(r)

    # ── Test 4 (optional): Live webcam ────────────────────────────────────────
    if args.live:
        print(f"\n{SEP}")
        print("TEST 4 — LIVE WEBCAM (real-time)")
        print(SEP)
        r = eval_live(model, device, n)
        if "avg" in r:
            print(f"  avg={r['avg']:.4f}  std={r['std']:.4f}  "
                  f"min={r['min']:.4f}  max={r['max']:.4f}")
        results.append(r)

    # ── Final report ──────────────────────────────────────────────────────────
    passed = print_report(results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
