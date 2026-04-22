# DeepShield — Demo Run Guide

## Quick Start (2-terminal setup)

### Terminal 1: Python Agent (webcam mode)
```powershell
cd deepfake
pip install -r requirements.txt
python -m agent.main
```

### Terminal 2: Electron Overlay
```powershell
cd deepfake\overlay
npm install
npm start
```

---

## Demo Mode A: Pre-recorded deepfake video (safest, recommended for hackathon)

```powershell
# Run agent pointing at the demo clip
python -m agent.main --demo demo\deepfake_sample.mp4
```

The agent will loop the video and run all detectors against it.
You should see the badge go RED within ~10-15 seconds as blink absence
and temporal jitter signals accumulate.

---

## Demo Mode B: OBS Virtual Camera (most impressive)

### Step 1 — Start OBS Virtual Camera
1. Open OBS Studio
2. Add a "Media Source" with your deepfake_sample.mp4
3. Start OBS Virtual Camera (Tools → Virtual Camera → Start)

### Step 2 — Play the deepfake through demo script
```powershell
python demo\play_deepfake.py --video demo\deepfake_sample.mp4
```

### Step 3 — Run agent on virtual camera device
```powershell
# Use --device 1 (or 2, depending on your system — try both)
python -m agent.main --device 1
```

### Step 4 — Start overlay
```powershell
cd overlay && npm start
```

---

## Finding your virtual camera device index

```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Device {i}: available")
    cap.release()
```

---

## Demo Flow Script (7 minutes)

1. **[0:00]** Show badge GREEN on your own face
2. **[1:00]** Explain: "Every blink, head movement, and pixel pattern is being checked"
3. **[2:00]** Switch to deepfake video source
4. **[2:10]** Badge turns YELLOW — "system is detecting anomalies"
5. **[2:30]** Badge turns RED — alert fires: "DEEPFAKE DETECTED"
6. **[3:00]** Click debug panel — show which signals triggered
7. **[4:00]** Architecture walkthrough
8. **[5:30]** Show SQLite audit log
9. **[6:00]** Switch back to real face — badge goes GREEN again
10. **[7:00]** Closing statement

---

## Checking the audit log

```powershell
# Install DB browser or use Python
python -c "
import sqlite3, json
conn = sqlite3.connect('data/audit.db')
for row in conn.execute('SELECT session_id, peak_risk, alert_count FROM sessions'):
    print(row)
for row in conn.execute('SELECT timestamp, trust_score, primary_trigger FROM alerts ORDER BY timestamp DESC LIMIT 5'):
    print(row)
"
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: mediapipe` | `pip install mediapipe` |
| Camera not found | Try `--device 0`, `1`, or `2` |
| Overlay doesn't connect | Make sure Python agent is running first |
| No face detected on video | Ensure face fills >20% of frame |
| `pyvirtualcam` install fails | Install OBS Studio first, then `pip install pyvirtualcam` |
