# 🧠 DEEPSHEILD — Real-Time AI Trust & Deepfake Detection System

> **A multi-layer intelligent system that verifies human authenticity in real-time and defends against deepfake, replay, and injection attacks.**

---

## 🚀 Live Demo

🔗 http://localhost:5173/ *(Replace with deployed link)*

---

## 🎯 Problem Statement

With the rapid rise of **deepfakes, AI-generated identities, and virtual camera injection (OBS attacks)**, traditional verification systems fail to distinguish:

* Real humans vs synthetic faces
* Live presence vs replay attacks
* Authentic identity vs manipulated identity

Most systems rely on **single-model detection**, making them fragile and easily bypassed.

---

## 💡 Solution — TruthLens

TruthLens is a **multi-signal AI verification engine** that combines:

* Behavioral analysis
* Liveness detection
* Identity verification
* Deepfake artifact detection
* Temporal consistency intelligence

👉 Instead of relying on one model, TruthLens makes **context-aware decisions**.

---

## 🧠 System Architecture

```text
Camera Input
     ↓
Face Detection
     ↓
┌──────────────────────────────┐
│ Multi-Signal Intelligence    │
│                              │
│ 1. GRU (Behavior)            │
│ 2. CNN (Liveness / Replay)   │
│ 3. Identity (ArcFace)        │
│ 4. Deepfake CNN              │
│ 5. Consistency Layer         │
└──────────────────────────────┘
     ↓
Fusion Engine (Risk-first logic)
     ↓
Final Decision (SAFE / WARNING / HIGH_RISK)
```

---

## 🔥 Key Features

### ✅ Real-Time Human Verification

* Detects natural behavior (blink, motion, micro-patterns)

### 🔁 Replay Attack Detection

* Identifies screen-based attacks (phone/laptop replays)

### 🎭 Deepfake Detection

* Trained on Celeb-DF + domain-aligned webcam data

### 🧬 Identity Verification

* Confirms same-person consistency using embeddings

### 🛡 OBS Injection Defense (🔥 Critical)

* Detects clean deepfake feeds from virtual cameras

### 🧠 Consistency Layer (Game-Changer)

* Identifies **unnatural temporal consistency**
* Catches attacks that bypass visual detection

---

## ⚙️ Tech Stack

### Backend

* Python
* PyTorch
* OpenCV
* MediaPipe

### Models

* GRU (Behavior Modeling)
* CNN (Replay Detection)
* EfficientNet-B0 (Deepfake Detection)
* ArcFace (Identity Verification)

### Frontend

* React + Vite
* TailwindCSS
* Framer Motion

---

## 🎯 System Capabilities

| Scenario             | Result       |
| -------------------- | ------------ |
| Real Webcam          | 🟢 SAFE      |
| Replay Attack        | 🔴 HIGH_RISK |
| Deepfake Video       | 🔴 HIGH_RISK |
| OBS Injection Attack | 🔴 HIGH_RISK |

---

## 🧪 Validation Highlights

* ✅ No false positives on real users
* ✅ Fast detection (~2–3s)
* ✅ Robust against clean deepfake injection
* ✅ Stable multi-signal fusion

---

## 🚨 Why This Is Different

Most solutions:

```text
Single Model → Binary Decision → Easy to Break
```

TruthLens:

```text
Multi-Signal System → Context-Aware Fusion → Hard to Bypass
```

---

## 🧠 Core Insight

> Deepfake detection alone is NOT enough.

TruthLens solves:

* Behavioral realism
* Identity consistency
* Temporal anomalies

👉 Making it **system-level intelligence**, not just a model.

---

## 📦 Project Structure

```text
agent/
 ├── ml/
 │   ├── gru_model.py
 │   ├── cnn_liveness.py
 │   ├── deepfake_efficientnet.py
 │   ├── identity_verifier.py
 │   └── fusion_engine.py
 │
 ├── input/
 │   └── webcam.py
 │
 ├── debug_ui.py
 └── main.py

scripts/
 ├── capture_real.py
 ├── capture_obs.py
 ├── train_deepfake_final.py
 └── test_deepfake_model.py
```

---

## ⚡ Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/truthlens.git
cd truthlens
```

### 2. Setup environment

```bash
python -m venv venv310
venv310\Scripts\activate
pip install -r requirements.txt
```

### 3. Run system

```bash
python -m agent.main --debug-ui
```

### 4. Run frontend

```bash
cd frontend
npm install
npm run dev
```

---

## 🎬 Demo Flow

1. Start system
2. Select mode
3. Show:

   * Real user → SAFE
   * Replay → HIGH_RISK
   * Deepfake → HIGH_RISK
   * OBS → HIGH_RISK

👉 Explain multi-layer detection

---

## 🏆 Hackathon Impact

* Real-world applicable
* Security-focused
* Hard-to-bypass architecture
* Strong demo storytelling

---

## 🔮 Future Improvements

* rPPG (heartbeat detection)
* Transformer-based temporal modeling
* Edge deployment optimization
* Mobile SDK integration

---

## 👨‍💻 Team

* Backend / AI System — You
* Frontend / UX — Team

---

## 📌 Final Note

TruthLens is not just a deepfake detector.

It is a:

> **Real-Time Human Authenticity Engine**

---

⭐ If you like this project, consider starring the repo!
