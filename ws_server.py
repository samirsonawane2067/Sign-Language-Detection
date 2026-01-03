# file: ws_server.py
"""
Stable WebSocket server for RealTimeDuplex.jsx + your KNN model
Run:
    python ws_server.py
"""

import asyncio
import json
from pathlib import Path
from collections import deque, Counter
from typing import Dict, Any, List, Deque, Optional

import numpy as np
import joblib
import websockets

# ------------------------------
# Paths
# ------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sign_knn.joblib"

# ------------------------------
# Prediction settings
# ------------------------------

PRED_SMOOTHING = 7
DEBOUNCE_FRAMES = 8
MIN_CONFIDENCE = 0.6

# ------------------------------
# Load Model
# ------------------------------

print("▶ Loading model:", MODEL_PATH)

if not MODEL_PATH.exists():
    print("❌ ERROR: Model not found!")
    print("Run:  python main.py train")
    exit(1)

payload = joblib.load(MODEL_PATH)
model = payload["model"]
LABELS = payload["labels"]

print("✔ Model loaded successfully.")
print("✔ Labels:", LABELS)


# -----------------------------------------------------------
# Convert LEFT/RIGHT hand of client → features for your KNN
# -----------------------------------------------------------
def frame_to_feature(frame: Dict[str, Any]) -> Optional[np.ndarray]:
    lh = frame.get("lh", [])
    rh = frame.get("rh", [])

    pts = None
    if lh:
        pts = np.array(lh, dtype=np.float32)
    elif rh:
        pts = np.array(rh, dtype=np.float32)
    else:
        return None

    # ensure 21 landmarks
    if pts.shape[0] < 21:
        pad = np.zeros((21 - pts.shape[0], 3), dtype=np.float32)
        pts = np.vstack([pts, pad])

    wrist = pts[0].copy()
    rel = pts - wrist
    denom = np.linalg.norm(pts[9] - wrist) + 1e-6
    rel /= denom

    return rel.flatten()


# -----------------------------------------------------------
# Predict label + confidence
# -----------------------------------------------------------
def predict_feat(feat: np.ndarray):
    idx = int(model.predict([feat])[0])
    conf = 1.0
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba([feat])[0].max())
    return LABELS[idx], conf


# -----------------------------------------------------------
# Map tokens → animation commands
# -----------------------------------------------------------
def map_tokens_to_avatar(tokens: List[str]):
    cmds = []
    t = 0
    for tok in tokens:
        cmds.append({
            "token": tok,
            "animation": f"{tok.lower()}_clip",
            "start_ms": t,
            "dur_ms": 600
        })
        t += 700
    return cmds


# -----------------------------------------------------------
# WebSocket Handler
# -----------------------------------------------------------
async def handler(ws, path):
    print("✔ Client connected")

    pred_window: Deque[str] = deque(maxlen=PRED_SMOOTHING)
    committed_tokens: List[str] = []
    last_committed = None
    stable = 0

    try:
        async for msg in ws:
            try:
                data = json.loads(msg)
            except:
                await ws.send(json.dumps({"type": "error", "msg": "invalid json"}))
                continue

            if data.get("type") != "landmark_sequence":
                continue

            frames = data.get("frames", [])

            for frame in frames:
                feat = frame_to_feature(frame)
                if feat is None:
                    pred_window.clear()
                    stable = 0
                    continue

                label, conf = predict_feat(feat)

                if conf < MIN_CONFIDENCE:
                    pred_window.clear()
                    stable = 0
                    continue

                pred_window.append(label)

                cnt = Counter(pred_window)
                maj, count = cnt.most_common(1)[0]

                if maj == last_committed:
                    stable = 0
                elif count >= 3:
                    stable += 1
                    if stable >= DEBOUNCE_FRAMES:
                        committed_tokens.append(maj)
                        last_committed = maj
                        stable = 0
                        pred_window.clear()

            # Prepare server response
            text = "".join(committed_tokens)
            avatar_cmds = map_tokens_to_avatar(committed_tokens)

            resp = {
                "type": "inference_result",
                "translated_text": text,
                "tokens": committed_tokens,
                "confidence": 0.9 if committed_tokens else 0.0,
                "avatar_commands": avatar_cmds,
            }

            await ws.send(json.dumps(resp))

            # clear tokens for next chunk
            committed_tokens = []

    except websockets.ConnectionClosed:
        print("✘ Client disconnected")


# -----------------------------------------------------------
# Start Server
# -----------------------------------------------------------
async def main():
    port = 8765
    print(f"\n🚀 WebSocket Server running at ws://0.0.0.0:{port}\n")
    async with websockets.serve(handler, "0.0.0.0", port, ping_interval=None):
        await asyncio.Future()  # keep running


if __name__ == "__main__":
    asyncio.run(main())
