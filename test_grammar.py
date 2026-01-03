# file: main.py
"""
Sign ↔ Voice Communication System
- Sign Language -> Text -> Voice
- Voice -> Text -> Sign Animation

Dependencies:
  pip install opencv-python mediapipe scikit-learn numpy joblib pillow pyttsx3 SpeechRecognition pyaudio

"""

import argparse
import time
from pathlib import Path
from collections import deque, Counter
from typing import Deque, List, Optional, Tuple
import threading
import queue  # For thread-safe communication

import cv2
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from gtts import gTTS
    import pygame
except ImportError:
    print("gTTS or pygame not installed. pip install gtts pygame")
    gTTS = None
    pygame = None

try:
    import socketio
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("python-socketio not installed. pip install python-socketio. WebSocket will be disabled.")

from PIL import Image, ImageSequence
import tempfile
import os

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = ROOT / "sign_assets"   # Folder with GIFs

MODEL_PATH = MODEL_DIR / "sign_knn.joblib"

WINDOW_TITLE = ""
PRED_SMOOTHING = 7
MIN_CONFIDENCE = 0.6
DEBOUNCE_FRAMES = 8
# Auto-send settings: send buffer automatically when number of non-space characters
# reaches this threshold, or if any buffer entry is a multi-character label (e.g., 'run').
DEFAULT_AUTO_SEND_THRESHOLD = 4

# -------------------------------
# Feature Extraction
# -------------------------------
# ... (HandFeatureExtractor class remains unchanged)
class HandFeatureExtractor:
    @staticmethod
    def _normalize(points: np.ndarray) -> np.ndarray:
        wrist = points[0]
        rel = points - wrist
        scale = np.linalg.norm(points[9] - wrist) + 1e-6
        rel /= scale
        return rel

    def from_mediapipe(self, hand_landmarks) -> np.ndarray:
        pts = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32)
        rel = self._normalize(pts)
        return rel.flatten()

# -------------------------------
# Data Collection
# -------------------------------
# ... (DataCollector class remains unchanged)
class DataCollector:
# ... (contents remain unchanged)
    def __init__(self, label: str, frames: int = 200, camera: int = 0, flip: bool = True):
        self.label = label
        self.frames = frames
        self.camera = camera
        self.flip = flip

    def run(self) -> None:
        if mp is None:
            raise RuntimeError("mediapipe not installed. pip install mediapipe")

        (DATA_DIR / self.label).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            raise RuntimeError("Cannot access webcam")

        # Welcome splash (1s)
        ok, welcome_frame = cap.read()
        if ok and self.flip:
            welcome_frame = cv2.flip(welcome_frame, 1)
        if ok:
            hh, ww = welcome_frame.shape[:2]
        else:
            hh, ww = 480, 640
        splash = np.zeros((hh, ww, 3), dtype=np.uint8)
        cv2.putText(splash, 'Welcome to Sign ↔ Voice', (int(ww*0.05), int(hh*0.45)), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(splash, 'Starting...', (int(ww*0.05), int(hh*0.55)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_TITLE, splash)
        cv2.waitKey(1000)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        extractor = HandFeatureExtractor()

        saved = 0
        target = self.frames
        start_t = time.time()

        while saved < target:
            ok, frame = cap.read()
            if not ok:
                break
            if self.flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                feats = extractor.from_mediapipe(res.multi_hand_landmarks[0])
                out_path = DATA_DIR / self.label / f"{int(time.time() * 1e6)}.npy"
                np.save(out_path, feats)
                saved += 1

            cv2.putText(frame, f"Collecting '{self.label}' {saved}/{target}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(WINDOW_TITLE, frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        print(f"Saved {saved} samples for '{self.label}' in {time.time()-start_t:.1f}s")

        hands.close()
        cap.release()
        cv2.destroyAllWindows()

# -------------------------------
# Trainer
# -------------------------------
# ... (Trainer class remains unchanged)
class Trainer:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X, y, labels = [], [], []
        for idx, label_dir in enumerate(sorted(p for p in DATA_DIR.iterdir() if p.is_dir())):
            labels.append(label_dir.name)
            for f in label_dir.glob("*.npy"):
                X.append(np.load(f))
                y.append(idx)
        if not X:
            raise RuntimeError("No data found. Use collect mode first.")
        return np.stack(X), np.array(y, dtype=np.int64), labels

    def train(self) -> None:
        X, y, labels = self.load_dataset()
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y)
        clf = KNeighborsClassifier(n_neighbors=min(7, len(np.unique(y))*2+1), weights="distance")
        clf.fit(X_tr, y_tr)

        preds = clf.predict(X_va)
        print(f"Validation accuracy: {accuracy_score(y_va, preds):.3f}")
        print(classification_report(y_va, preds, target_names=labels))

        joblib.dump({"model": clf, "labels": labels}, self.model_path)
        print(f"Saved model to {self.model_path}")

# -------------------------------
# Helper: Show GIF Animation
# -------------------------------
# ... (text_to_sign_animation function remains unchanged)
def text_to_sign_animation(text: str, frame: np.ndarray) -> np.ndarray:
    for ch in text:
        gif_path = ASSETS_DIR / f"{ch.lower()}.gif"
        if not gif_path.exists():
            gif_path = ASSETS_DIR / f"{ch.upper()}.gif"
        if gif_path.exists():
            pil_gif = Image.open(gif_path)
            frames = [cv2.cvtColor(np.array(f.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
                      for f in ImageSequence.Iterator(pil_gif)]
            for gif_frame in frames:
                # Resize GIF to fit
                h, w = gif_frame.shape[:2]
                scale = 0.5
                gif_frame = cv2.resize(gif_frame, (int(w*scale), int(h*scale)))
                fh, fw = gif_frame.shape[:2]
                # Overlay top-left corner
                frame[0:fh, 0:fw] = gif_frame
                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(40) & 0xFF == 27:
                    break
    return frame


# -------------------------------
# Real-time Recognizer
# -------------------------------
class RealtimeRecognizer:
    def __init__(self, model_path: Path = MODEL_PATH, camera: int = 0, flip: bool = True, per_letter_tts: bool = False, full_duplex: bool = False, auto_send_threshold: int = DEFAULT_AUTO_SEND_THRESHOLD):
        self.camera = camera
        self.flip = flip
        self.per_letter_tts = per_letter_tts
        self.full_duplex = full_duplex
        self.auto_send_threshold = auto_send_threshold if isinstance(auto_send_threshold, int) and auto_send_threshold > 0 else DEFAULT_AUTO_SEND_THRESHOLD
        self.buffer: List[str] = []
        self.pred_window: Deque[str] = deque(maxlen=PRED_SMOOTHING)
        self.last_committed: Optional[str] = None
        self.stable_count = 0
        # UI state
        self.current_letter: Optional[str] = None
        self.last_confidence: float = 0.0
        # Hold duration tracking
        self.letter_hold_start_time: Optional[float] = None
        self.letter_added_once: bool = False
        self.extra_letters_added: int = 0
        
        # Initialize WebSocket client for full-duplex mode
        self.sio = None
        if full_duplex and SOCKETIO_AVAILABLE:
            try:
                import socketio as sio_module
                self.sio = sio_module.Client()
                self.sio.on('connect', self._on_ws_connect)
                self.sio.on('disconnect', self._on_ws_disconnect)
                self.sio.on('speak', self._on_ws_speak)
                self.sio.on('status', self._on_ws_status)
                print("[WebSocket] Attempting to connect to ws://localhost:5000...")
                # Connect to the WebSocket server (websocket_server.py) on port 5000
                self.sio.connect('http://localhost:5000', transports=['websocket'])
                # We'll register when the 'connect' event is received to ensure we're connected
                self._sio_registered = False
            except Exception as e:
                print(f"[WebSocket ERROR] Could not connect: {e}")
                self.sio = None
        
        # Initialize pygame mixer for audio playback
        if pygame:
            try:
                pygame.mixer.init()
            except Exception as e:
                print(f"[PYGAME INIT WARNING] {e}")

        payload = joblib.load(model_path)
        self.model: KNeighborsClassifier = payload["model"]
        self.labels: List[str] = payload["labels"]

    def _on_ws_connect(self):
        print("[WebSocket] ✓ Connected to server")
        print(f"[WebSocket] Socket ID: {self.sio.sid}")
        # Register this client as the Python recognizer
        if not getattr(self, '_sio_registered', False):
            try:
                self.sio.emit('register_client', {'type': 'python'})
                self._sio_registered = True
                print('[WebSocket] ✓ Sent register_client (type=python)')
            except Exception as e:
                print(f'[WebSocket] ✗ Register failed: {e}; trying python_ready fallback')
                try:
                    self.sio.emit('python_ready', {})
                    self._sio_registered = True
                    print('[WebSocket] ✓ Sent python_ready (fallback)')
                except Exception as e2:
                    print(f'[WebSocket] ✗ python_ready fallback failed: {e2}')
        # Log the server status if available
        try:
            self.sio.emit('get_status', {})
        except Exception:
            pass

    def _on_ws_status(self, data):
        print('[WebSocket] Server status event:', data)

    def _on_ws_disconnect(self):
        print("[WebSocket] Disconnected from server")

    def _on_ws_speak(self, data):
        """WebSocket callback when web client sends text for TTS"""
        text = data.get('text', '')
        if text:
            print(f"[WebSocket] Speaking from web: {text}")
            self._speak(text)

    # --- THREADING ACTION FUNCTION (Uses gTTS + pygame) ---
    def _speak_action(self, text: str):
        """Execute TTS using gTTS and pygame in a separate thread"""
        if not text or not gTTS or not pygame:
            return

        try:
            print(f"[TTS] Speaking: {text}")
            # Create gTTS object
            tts = gTTS(text=text, lang="en", slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                temp_file = fp.name
                tts.save(fp.name)
            
            # Play using pygame
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pass
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"[TTS ERROR] {e}")

    # --- NON-BLOCKING _speak FUNCTION (Starts the thread) ---
    def _speak(self, text: str) -> None:
        """Speak text without blocking the main video loop"""
        if not text:
            return
        
        # Run the blocking speech call in a separate thread
        speech_thread = threading.Thread(target=self._speak_action, args=(text,), daemon=True)
        speech_thread.start()

    def _predict_label(self, feats: np.ndarray) -> Tuple[str, float]:
        pred_idx = self.model.predict([feats])[0]
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([feats])[0]
            conf = float(np.max(probs))
        else:
            conf = 1.0
        return self.labels[pred_idx], conf

    def _show_welcome_window(self) -> None:
        """Display a welcome window with image for 2 seconds before starting the camera"""
        try:
            # Try to load the welcome image
            welcome_image_path = Path(__file__).parent / "welcome_image.png"
            
            if welcome_image_path.exists():
                # Load and display the welcome image for 2 seconds
                welcome_img = cv2.imread(str(welcome_image_path))
                if welcome_img is not None:
                    h, w = welcome_img.shape[:2]
                    cv2.imshow(WINDOW_TITLE, welcome_img)
                    cv2.waitKey(2000)  # Display for 2 seconds
                    return
            
            # Fallback: Create a text-based welcome screen if image not found
            cap = cv2.VideoCapture(self.camera)
            if not cap.isOpened():
                return
            
            ok, frame = cap.read()
            if ok and self.flip:
                frame = cv2.flip(frame, 1)
            
            if ok:
                h, w = frame.shape[:2]
            else:
                h, w = 480, 640
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Create welcome screen
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Add welcome text
            cv2.putText(frame, 'Welcome to Sign Voice', (int(w*0.1), int(h*0.35)),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
            
            cv2.putText(frame, 'Communication System', (int(w*0.1), int(h*0.45)),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
            
            # Add instructions
            cv2.putText(frame, 'Make hand gestures to recognize signs', (int(w*0.1), int(h*0.60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
            
            cv2.putText(frame, "Press 's' for space, Space/Enter to speak", (int(w*0.1), int(h*0.68)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
            
            cv2.putText(frame, 'Press any key to start...', (int(w*0.1), int(h*0.80)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow(WINDOW_TITLE, frame)
            cv2.waitKey(0)  # Wait for any key press
            
            cap.release()
        except Exception as e:
            print(f"[WELCOME] Error showing welcome window: {e}")

    def _voice_to_sign(self, frame: np.ndarray) -> np.ndarray:
        import speech_recognition as sr
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                cv2.putText(frame, "Listening...", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(WINDOW_TITLE, frame)
                cv2.waitKey(1)

                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            spoken_text = r.recognize_google(audio)
            print(f"[VOICE] Recognized: {spoken_text}")

            cv2.putText(frame, f"Recognized: {spoken_text}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            frame = text_to_sign_animation(spoken_text, frame)
            return frame

        except sr.UnknownValueError:
            msg = "Could not understand audio"
            print(f"[VOICE] {msg}")
            cv2.putText(frame, msg, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        except Exception as e:
            msg = f"VOICE ERROR: {e}"
            print(f"[VOICE ERROR] {e}")
            cv2.putText(frame, msg, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

    def run(self) -> None:
        if mp is None:
            raise RuntimeError("mediapipe not installed. pip install mediapipe")

        # Show welcome window before starting camera
        self._show_welcome_window()

        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            raise RuntimeError("Cannot access webcam")

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        extractor = HandFeatureExtractor()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if self.flip:
                frame = cv2.flip(frame, 1)

            # Draw header and status UI
            h, w = frame.shape[:2]
            # Show screen size (width x height)
            screen_text = f'{w}x{h}'
            
            # Commented out the gray header background
            # cv2.rectangle(frame, (0,0), (w, 100), (50,50,50), -1)
            # Header background
            
            # Prediction and accuracy display (use last_confidence as realtime accuracy indicator)
            # We'll render accuracy below the 'Pred' line and above the 'Text' area
            conf_pct = self.last_confidence * 100.0
            acc_text = f'Accuracy: {conf_pct:0.2f}%'

            # Draw controls on the right side
            control_y = 40
            control_x = w - 250
            # Controls header
            cv2.putText(frame, "CONTROLS:", (control_x, control_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            # Control items
            controls = [
                ("'S' key", "Add space", (0, 0, 255)),
                ("Space/Enter", "  Speak text", (0, 200, 200)),
                ("Backspace", "Delete last", (200, 0, 0)),
            ]
            for i, (key, action, color) in enumerate(controls, 1):
                y = control_y + (i * 30)
                cv2.putText(frame, f"{key}:", (control_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{action}", (control_x + 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

            # Draw text buffer area below header
            text_area_y = 120
            # Draw label 'Text:' in green (keep it highlighted)
            label_y = text_area_y + 8
            cv2.putText(frame, 'Text:', (20, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

            # render buffer tokens with colors (letters in green, space as gap)
            x_cursor = 90
            y_cursor = text_area_y + 8
            for token in self.buffer:
                tok = str(token)
                if tok == ' ':
                    # space token - just add spacing, don't draw anything
                    x_cursor += 15
                else:
                    # normal letter in green
                    cv2.putText(frame, tok.upper(), (x_cursor, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
                    x_cursor += 18

            cv2.putText(frame, f"Pred: {self.current_letter or '-'}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # Draw accuracy below the rendered text buffer
            acc_y = y_cursor + 40
            cv2.putText(frame, acc_text, (20, acc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)

            # Process hand landmarks with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            # Process hand landmarks if any
            if res.multi_hand_landmarks:
                # Draw hand landmarks on frame
                for hand_landmarks in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                
                feats = extractor.from_mediapipe(res.multi_hand_landmarks[0])
                pred_label, conf = self._predict_label(feats)
                # store last confidence for UI
                self.last_confidence = conf
                if conf < MIN_CONFIDENCE:
                    pred_label = None
            else:
                pred_label = None

            if pred_label:
                self.pred_window.append(pred_label)
                maj_label, maj_count = Counter(self.pred_window).most_common(1)[0]
                if maj_count >= max(3, PRED_SMOOTHING // 2):
                    self.stable_count += 1
                    
                    # First time detecting this letter
                    if self.stable_count == DEBOUNCE_FRAMES:
                        # Initialize hold tracking
                        self.letter_hold_start_time = time.time()
                        self.letter_added_once = False
                        self.extra_letters_added = 0
                        self.last_committed = maj_label
                        self.current_letter = maj_label
                    
                    # Add letter only once on first detection
                    if self.stable_count == DEBOUNCE_FRAMES and not self.letter_added_once:
                        self.buffer.append(maj_label)
                        self.letter_added_once = True
                        self._last_buffer_activity = time.time()
                        
                        # Send recognized letter to web in real-time
                        if self.full_duplex and self.sio and self.sio.connected:
                            try:
                                self.sio.emit('sign_recognized', {'text': maj_label})
                                print(f"[WebSocket] Sent letter: '{maj_label}'")
                            except Exception as e:
                                print(f"[WebSocket] Could not send letter: {e}")
                        
                        if self.per_letter_tts:
                            self._speak(maj_label)
                    
                    # Check for hold duration to add extra letters
                    elif self.letter_added_once and self.letter_hold_start_time:
                        hold_time = time.time() - self.letter_hold_start_time
                        
                        # Add second letter at 2 seconds
                        if hold_time > 2.0 and self.extra_letters_added == 0:
                            self.buffer.append(maj_label)
                            self.extra_letters_added = 1
                            self._last_buffer_activity = time.time()
                            
                            if self.full_duplex and self.sio and self.sio.connected:
                                try:
                                    self.sio.emit('sign_recognized', {'text': maj_label})
                                    print(f"[WebSocket] Sent extra letter (2s): '{maj_label}'")
                                except Exception as e:
                                    pass
                        
                        # Add third letter at 4 seconds
                        elif hold_time > 4.0 and self.extra_letters_added == 1:
                            self.buffer.append(maj_label)
                            self.extra_letters_added = 2
                            self._last_buffer_activity = time.time()
                            
                            if self.full_duplex and self.sio and self.sio.connected:
                                try:
                                    self.sio.emit('sign_recognized', {'text': maj_label})
                                    print(f"[WebSocket] Sent extra letter (4s): '{maj_label}'")
                                except Exception as e:
                                    pass
            else:
                # Hand gesture ended, reset tracking
                self.pred_window.clear()
                self.stable_count = 0
                self.letter_hold_start_time = None
                self.letter_added_once = False
                self.extra_letters_added = 0

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32 or key == 13:  # Space or Enter → speak sentence
                text = "".join(self.buffer).strip()
                if text:
                    # Send full sentence to web via WebSocket (if connected)
                    if self.full_duplex and self.sio and self.sio.connected:
                        try:
                            self.sio.emit('sign_recognized', {'text': text})
                            print(f"[WebSocket] Sent full sentence: '{text}'")
                        except Exception as e:
                            print(f"[WebSocket] Could not send full sentence: {e}")
                    self._speak(text)  # Non-blocking call
                self.last_committed = None
            elif key == 8:  # Backspace
                if self.buffer:
                    self.buffer.pop()
                    self._last_buffer_activity = time.time()
            elif key == ord("s") or key == ord("S"):
                self.buffer.append(" ")
                self.last_committed = None
                self._last_buffer_activity = time.time()
            elif key == ord("v") or key == ord("V"):
                frame = self._voice_to_sign(frame)

            # Auto-send full sentence after a short pause (if full_duplex is enabled)
            if self.full_duplex and self.sio and self.sio.connected and len(self.buffer) > 0:
                last_activity = getattr(self, '_last_buffer_activity', None)
                if last_activity and (time.time() - last_activity) > 1.2:
                    # Only auto-send if we've reached the config threshold OR we have
                    # a multi-letter buffer element (assumed whole word was recognized).
                    non_space_chars = len([c for c in self.buffer if c != " "])
                    contains_word = any(len(str(elem).strip()) > 1 for elem in self.buffer)
                    if non_space_chars >= self.auto_send_threshold or contains_word:
                        # Send the current buffer as a sentence
                        text = "".join(self.buffer).strip()
                        if text:
                            try:
                                self.sio.emit('sign_recognized', {'text': text})
                                print(f"[WebSocket] Auto-sent full sentence: '{text}'")
                            except Exception as e:
                                print(f"[WebSocket] Could not auto-send: {e}")
                        # Reset buffer after send
                        self.buffer = []
                        self._last_buffer_activity = None

        hands.close()
        cap.release()
        cv2.destroyAllWindows()


# -------------------------------
# CLI Parser
# -------------------------------
# ... (parse_args function remains unchanged)
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sign ↔ Voice App")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect")
    pc.add_argument("--label", required=True)
    pc.add_argument("--frames", type=int, default=200)
    pc.add_argument("--camera", type=int, default=0)
    pc.add_argument("--no_flip", action="store_true")

    sub.add_parser("train")

    pr = sub.add_parser("run")
    pr.add_argument("--camera", type=int, default=0)
    pr.add_argument("--per_letter_tts", action="store_true")
    pr.add_argument("--full_duplex", action="store_true", help="Enable WebSocket two-way mode with web avatar")
    pr.add_argument("--auto_send_threshold", type=int, default=DEFAULT_AUTO_SEND_THRESHOLD,
                    help="Auto-send the sign buffer when N non-space characters or a multi-char label are collected")
    pr.add_argument("--no_flip", action="store_true")

    return p.parse_args(argv)


# -------------------------------
# Main Entry
# -------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.cmd == "collect":
        DataCollector(args.label.lower(), args.frames, args.camera, flip=not args.no_flip).run()
    elif args.cmd == "train":
        Trainer().train()
    elif args.cmd == "run":
        if not MODEL_PATH.exists():
            raise SystemExit("Model not found. Run train first.")
        RealtimeRecognizer(camera=args.camera, flip=not args.no_flip, per_letter_tts=args.per_letter_tts, 
                  full_duplex=args.full_duplex, auto_send_threshold=args.auto_send_threshold).run()


if __name__ == "__main__":
    main()
