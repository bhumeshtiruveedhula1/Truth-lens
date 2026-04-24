"""
deepfake_model.py
=================
Clean, self-contained inference module extracted from the HongguLiu/Deepfake-Detection
repository (https://github.com/HongguLiu/Deepfake-Detection).

AUDIT-VERIFIED BEHAVIOR — nothing here deviates from the original inference path:
  - Same Xception backbone (network/xception.py)
  - Same TransferModel wrapper (network/models.py)
  - Same preprocessing pipeline (dataset/transform.py → xception_default_data_transforms['test'])
  - Same weight loading with DataParallel-key stripping
  - Same BGR→RGB→PIL→Resize(299)→ToTensor→Normalize([0.5],[0.5]) order
  - Same Softmax post-processing; returns probs[0][1] (fake probability)

Label convention (from original repo):
  0 = real
  1 = fake (deepfake)

Input:  numpy image in BGR format (as returned by OpenCV)
Output: float in [0.0, 1.0]  — probability that the image is a deepfake

Usage:
    from deepfake_model import load_model, predict

    model = load_model("./pretrained_model/df_c0_best.pkl")
    prob  = predict(face_crop_bgr, model)
    # prob → e.g. 0.93 means 93% likely deepfake
"""

import cv2
import torch
import torch.nn as nn
from PIL import Image as pil_image
from torchvision import transforms

# ── Re-import only the model construction code (no training/dataset deps) ─────
from network.models import model_selection


# ─────────────────────────────────────────────────────────────────────────────
# Constants — verbatim from dataset/transform.py and xception.py docstring
# ─────────────────────────────────────────────────────────────────────────────

INPUT_SIZE   = 299                        # Xception requires 299×299
MEAN         = [0.5, 0.5, 0.5]           # from xception_default_data_transforms
STD          = [0.5, 0.5, 0.5]           # pixel range after norm: [-1, +1]
NUM_CLASSES  = 2                          # 0=real, 1=fake
DROPOUT      = 0.5                        # must match checkpoint architecture
MODEL_CHOICE = 'xception'                 # primary inference backbone

# Exact test transform from dataset/transform.py (xception_default_data_transforms['test'])
_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Softmax post-function — matches predict_with_model() in detect_from_video.py
_SOFTMAX = nn.Softmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path: str, device: str = None) -> nn.Module:
    """
    Construct the TransferModel(xception) and load task-specific weights.

    This exactly mirrors the loading sequence in detect_from_video.py and
    test_CNN.py, including the DataParallel key-stripping guard.

    Args:
        weights_path: Path to the .pth / .pkl checkpoint file.
                      Download from: https://drive.google.com/drive/folders/1GNtk3hLq6sUGZCGx8fFttvyNYH8nrQS8
        device:       'cpu', 'cuda', or None (auto-detected).

    Returns:
        Loaded model in eval() mode, moved to the selected device.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Construct model — must match architecture used during training
    # model_selection returns TransferModel with:
    #   Xception backbone → Dropout(0.5) → Linear(2048, 2) as last_linear
    model = model_selection(
        modelname=MODEL_CHOICE,
        num_out_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )

    # Load checkpoint — use map_location so CPU-only machines don't error
    state_dict = torch.load(weights_path, map_location=device)

    # Strip DataParallel 'module.' prefix if weights were saved with nn.DataParallel
    # (mirrors the guard in detect_from_video.py and test_CNN.py)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()  # disable Dropout at inference time

    return model


def preprocess(image_bgr) -> torch.Tensor:
    """
    Convert a BGR numpy image (OpenCV format) into the exact tensor expected
    by the Xception model.

    Pipeline (verbatim from detect_from_video.py → preprocess_image +
    dataset/transform.py → xception_default_data_transforms['test']):

        BGR ndarray
          → cv2.cvtColor(BGR→RGB)
          → PIL.Image.fromarray()
          → Resize(299, 299)
          → ToTensor()          [0,255] uint8 → [0.0, 1.0] float
          → Normalize([0.5]*3, [0.5]*3)  → range [-1, 1]
          → unsqueeze(0)        add batch dim

    Args:
        image_bgr: numpy array, shape (H, W, 3), dtype uint8, BGR channel order.

    Returns:
        torch.Tensor of shape [1, 3, 299, 299], float32.
        NOTE: returned tensor is on CPU. Move to the model's device before forward pass.
    """
    # Step 1: BGR → RGB  (exact line from detect_from_video.py)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Step 2: numpy → PIL  (required for torchvision transforms to behave identically)
    pil_img = pil_image.fromarray(image_rgb)

    # Step 3: Resize → ToTensor → Normalize  (xception_default_data_transforms['test'])
    tensor = _TEST_TRANSFORM(pil_img)  # shape [3, 299, 299]

    # Step 4: Add batch dimension  (mirrors preprocessed_image.unsqueeze(0))
    tensor = tensor.unsqueeze(0)       # shape [1, 3, 299, 299]

    return tensor


def predict(image_bgr, model: nn.Module) -> float:
    """
    Run deepfake detection on a single face crop.

    This mirrors predict_with_model() from detect_from_video.py but returns
    a clean float probability instead of (label, softmax_tensor).

    Args:
        image_bgr: numpy array (H, W, 3) uint8, BGR format (e.g., from cv2).
                   Should be a face crop — not a full frame.
        model:     Loaded model returned by load_model().

    Returns:
        deepfake_probability (float): value in [0.0, 1.0].
            → 0.0 = confidently real
            → 1.0 = confidently fake (deepfake)

    Example:
        model = load_model("./pretrained_model/df_c0_best.pkl")
        prob  = predict(face_crop_bgr, model)
        print(f"Deepfake probability: {prob:.2%}")
    """
    # Determine which device the model lives on
    device = next(model.parameters()).device

    # Preprocess: BGR numpy → [1, 3, 299, 299] float tensor
    tensor = preprocess(image_bgr).to(device)

    # Forward pass — no gradient needed for inference
    with torch.no_grad():
        logits = model(tensor)               # shape [1, 2], raw logits

    # Softmax → class probabilities (sums to 1.0)
    # Exact post_function from detect_from_video.py: nn.Softmax(dim=1)
    probs = _SOFTMAX(logits)                 # shape [1, 2]

    # probs[0][0] = P(real), probs[0][1] = P(fake)
    deepfake_probability = float(probs[0][1].cpu().item())

    return deepfake_probability


# ─────────────────────────────────────────────────────────────────────────────
# Example usage (run as script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import numpy as np

    WEIGHTS = './pretrained_model/df_c0_best.pkl'

    # --- Load model -----------------------------------------------------------
    print(f"[deepfake_model] Loading model from: {WEIGHTS}")
    try:
        model = load_model(WEIGHTS)
        print(f"[deepfake_model] Model loaded on: {next(model.parameters()).device}")
    except FileNotFoundError:
        print(f"[deepfake_model] ERROR: Weights not found at '{WEIGHTS}'")
        print("  Download from: https://drive.google.com/drive/folders/1GNtk3hLq6sUGZCGx8fFttvyNYH8nrQS8")
        sys.exit(1)

    # --- Predict on an image file or a random tensor (demo) -------------------
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"[deepfake_model] Running inference on: {image_path}")
        face_crop = cv2.imread(image_path)
        if face_crop is None:
            print(f"[deepfake_model] ERROR: Could not read image at '{image_path}'")
            sys.exit(1)
    else:
        # Synthetic test: random BGR noise image (results will be meaningless)
        print("[deepfake_model] No image path provided — using synthetic random face crop.")
        face_crop = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    prob = predict(face_crop, model)

    print(f"\n  Deepfake probability : {prob:.4f}  ({prob:.1%})")
    print(f"  Verdict              : {'FAKE' if prob >= 0.5 else 'REAL'}")
