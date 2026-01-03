Full-Duplex Real-Time Sign ↔ Voice Translation Setup

This document explains how to run the complete two-way system with WebSocket.

Prerequisites
- Python 3.9+
- pip packages (all): opencv-python mediapipe scikit-learn numpy joblib gTTS pygame SpeechRecognition python-socketio flask-socketio

Install WebSocket packages
```
pip install python-socketio flask-socketio python-engineio
```

Quick Start — Three Terminal Windows

Terminal 1: Start WebSocket Server
```
cd "C:\Users\samir\OneDrive\Desktop\3.0"
pip install flask flask-socketio python-socketio
python websocket_server.py
```
Output: Server listening on ws://localhost:5000

Terminal 2: Start Python Sign Recognizer (with full-duplex)
```
cd "C:\Users\samir\OneDrive\Desktop\3.0"
python main.py run --full_duplex
```
This will:
- Recognize sign language from webcam (text → OpenCV + MediaPipe)
- Send recognized text to WebSocket server
- Receive text from web client and play TTS

Terminal 3: Start Web Avatar Server
```
cd "C:\Users\samir\OneDrive\Desktop\3.0\web"
python -m http.server 8000
```
Open browser: http://localhost:8000

Flow (Full-Duplex Mode)
1. User signs → Python recognizes text → sends to WebSocket
2. Web demo receives text → automatically animates avatar → shows on-screen
3. User types in web UI → clicks "Send to Python" → Python receives text and plays TTS
4. WebSocket indicator shows connection status (green = connected, red = disconnected)

Modes

Solo Sign Recognition (no web)
```
python main.py run
```
- Recognizes sign language
- Speaks via TTS when you press Space
- No WebSocket needed

Solo Web Avatar (no Python)
```
cd web; python -m http.server 8000
# Open http://localhost:8000
```
- Manual avatar animation via text input
- Type word and press "Animate"

Full-Duplex (Python + Web + WebSocket)
- Start all three terminals as shown above
- Python feed recognized sign → web animates in real-time
- Web can send text back to Python for TTS

Troubleshooting

WebSocket won't connect
- Check WebSocket server is running (Terminal 1 output)
- Ensure no firewall blocking localhost:5000
- Check console in browser (F12) for WebSocket errors

Avatar animations not triggering
- Console (F12) should show "animate_text" messages from Python
- Check that web/models/avatar.glb exists and is loaded (Console should show "GLTF clips: ...")
- Verify Fox.glb has clips: Survey, Walk, Run

Python recognizer can't find model
- Run `python main.py train` first to create the model
- Collect training data: `python main.py collect --label a --frames 200` (repeat for a-z)

Tips
- WebSocket is optional; the system works without it (just use `python main.py run`)
- Customize animation mapping in `web/app.js` (textToAnimMap)
- Replace `web/models/avatar.glb` with your own Blender GLB export
