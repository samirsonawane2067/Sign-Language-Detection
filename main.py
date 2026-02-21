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
import json
from datetime import datetime
import nltk
from nltk.corpus import words

import cv2
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("tkinter not available for screen size detection")

# Import grammar corrector with API support
from grammar_corrector import correct_grammar, initialize_corrector

# Initialize grammar corrector with OpenAI enabled
initialize_corrector(use_transformer=True, use_openai=True, use_google_nlp=False)

# Import performance metrics
from performance_metrics import PerformanceMetrics

# Import emotion detector
from emotion_detector import EmotionDetector

# Import multi-language support
from multi_language_support import LanguageManager, MultiLanguageRecognizer

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
try:
    import requests
except ImportError:
    requests = None

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = ROOT / "sign_assets"   # Folder with GIFs
HISTORY_FILE = ROOT / "history.json"

MODEL_PATH = MODEL_DIR / "sign_knn.joblib"

WINDOW_TITLE = "Sign Language Recognition"
PRED_SMOOTHING = 7
MIN_CONFIDENCE = 0.15  # Further lowered to catch more predictions
MIN_CONFIDENCE_ISL = 0.1  # ISL even more lenient

DEBOUNCE_FRAMES = 3  # Reduced from 8 for faster responsiveness
# Auto-send settings: send buffer automatically when number of non-space characters
# reaches this threshold, or if any buffer entry is a multi-character label (e.g., 'run').
DEFAULT_AUTO_SEND_THRESHOLD = None  # Disabled - no auto-send limit

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
        hands = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lowered from default 0.5
            min_tracking_confidence=0.3    # Lowered from default 0.5
        )
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
    import cv2
    import numpy as np
    from PIL import Image, ImageSequence
    
    for ch in text:
        # Try MP4 first, then GIF as fallback
        video_path = ASSETS_DIR / f"{ch.lower()}.mp4"
        if not video_path.exists():
            video_path = ASSETS_DIR / f"{ch.upper()}.mp4"
        if not video_path.exists():
            # Fallback to GIF if MP4 not found
            gif_path = ASSETS_DIR / f"{ch.lower()}.gif"
            if not gif_path.exists():
                gif_path = ASSETS_DIR / f"{ch.upper()}.gif"
            if gif_path.exists():
                pil_gif = Image.open(gif_path)
                frames = [cv2.cvtColor(np.array(f.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
                          for f in ImageSequence.Iterator(pil_gif)]
                for gif_frame in frames:
                    # Resize GIF to fit - increased size
                    h, w = gif_frame.shape[:2]
                    scale = 0.8  # Increased from 0.5 to 0.8 for larger display
                    gif_frame = cv2.resize(gif_frame, (int(w*scale), int(h*scale)))
                    fh, fw = gif_frame.shape[:2]
                    
                    margin_x, margin_y = 20, 130
                    # Ensure GIF fits within frame bounds
                    max_y, max_x = frame.shape[:2]
                    if fh + margin_y > max_y:
                        scale = (max_y - margin_y) / h
                        gif_frame = cv2.resize(gif_frame, (int(w*scale), int(h*scale)))
                        fh, fw = gif_frame.shape[:2]
                    if fw + margin_x > max_x:
                        scale = (max_x - margin_x) / w
                        gif_frame = cv2.resize(gif_frame, (int(w*scale), int(h*scale)))
                        fh, fw = gif_frame.shape[:2]
                    
                    # Overlay GIF in top-left corner with margin
                    frame[margin_y:margin_y+fh, margin_x:margin_x+fw] = gif_frame
                    cv2.imshow(WINDOW_TITLE, frame)
                    if cv2.waitKey(40) & 0xFF == 27:
                        break
                return frame
        
        # Handle MP4 video
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                while True:
                    ret, video_frame = cap.read()
                    if not ret:
                        cap.release()
                        break
                    
                    # Resize video frame to fit - increased size
                    h, w = video_frame.shape[:2]
                    scale = 1.2  # Increased from 0.8 to 1.2 for larger display
                    video_frame = cv2.resize(video_frame, (int(w*scale), int(h*scale)))
                    fh, fw = video_frame.shape[:2]
                    
                    # Position video in top-left corner with some margin
                    margin_x, margin_y = 20, 130  # Moved down from 20 to 60 for spoken sentence space
                    # Ensure video fits within frame bounds
                    max_y, max_x = frame.shape[:2]
                    if fh + margin_y > max_y:
                        scale = (max_y - margin_y) / h
                        video_frame = cv2.resize(video_frame, (int(w*scale), int(h*scale)))
                        fh, fw = video_frame.shape[:2]
                    if fw + margin_x > max_x:
                        scale = (max_x - margin_x) / w
                        video_frame = cv2.resize(video_frame, (int(w*scale), int(h*scale)))
                        fh, fw = video_frame.shape[:2]
                    
                    # Overlay video in top-left corner with margin
                    frame[margin_y:margin_y+fh, margin_x:margin_x+fw] = video_frame
                    cv2.imshow(WINDOW_TITLE, frame)
                    
                    # Play at appropriate speed (30fps)
                    if cv2.waitKey(33) & 0xFF == 27:
                        cap.release()
                        break
                cap.release()
                continue  # Move to next character
    
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
        
        # Grammar correction display
        self.corrected_display_text: Optional[str] = None
        self.display_correction_until: Optional[float] = None
        
        # Voice input flag
        self.voice_input_requested: bool = False
        
        # Emotion detection
        self.emotion_detector = None
        self.current_emotion: str = "neutral"
        self.emotion_confidence: float = 0.0
        try:
            self.emotion_detector = EmotionDetector()
            print("[EMOTION] Emotion detector initialized")
        except Exception as e:
            print(f"[EMOTION] Could not initialize emotion detector: {e}")
        
        # Load the trained ISL model
        self.isl_model = None
        try:
            self.isl_model = joblib.load('models/sign_knn_isl_dual_hands.joblib')
            print("[ISL] ISL dual hands model loaded successfully")
            print(f"[ISL] Classes available: {list(self.isl_model.classes_)}")
        except Exception as e:
            print(f"[ISL] ISL model not found or could not load: {e}")
            self.isl_model = None
        
        # Initialize WebSocket client for full-duplex mode
        self.sio = None
        if full_duplex and SOCKETIO_AVAILABLE:
            try:
                import socketio as sio_module
                self.sio = sio_module.Client(reconnection=True)
                self.sio.on('connect', self._on_ws_connect)
                self.sio.on('disconnect', self._on_ws_disconnect)
                self.sio.on('speak', self._on_ws_speak)
                self.sio.on('status', self._on_ws_status)
                print("[WebSocket] Attempting to connect to ws://localhost:5000...")
                # Connect to the WebSocket server (websocket_server.py) on port 5000
                try:
                    # Prefer default transport negotiation (polling -> websocket)
                    self.sio.connect('http://localhost:5000')
                except Exception as e_primary:
                    print(f"[WebSocket] Primary connect failed ({e_primary}); trying websocket-only fallback...")
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

        # Initialize multi-language support (loads both ASL and ISL models)
        self.language_manager = LanguageManager(ROOT)
        self.ml_recognizer = MultiLanguageRecognizer(self.language_manager)
        
        # Fallback: load old model path if multi-language models not available
        try:
            self.model: Optional[KNeighborsClassifier] = None
            self.labels: List[str] = []
            
            # Check if we have at least one language model loaded
            if not self.ml_recognizer.models:
                print("[WARNING] No multi-language models found, attempting fallback to legacy model...")
                try:
                    payload = joblib.load(model_path)
                    self.model = payload["model"]
                    self.labels = payload["labels"]
                    print(f"[LEGACY] Loaded legacy model from {model_path}")
                except Exception as e:
                    print(f"[ERROR] Could not load legacy model: {e}")
                    raise RuntimeError("No models available for recognition")
            else:
                print(f"[LANGUAGE] Multi-language recognizer initialized with models: {list(self.ml_recognizer.models.keys())}")
                # Get labels from current language
                model_info = self.ml_recognizer.get_model_info()
                self.labels = model_info.get("labels", [])
        except Exception as e:
            print(f"[MODEL INIT ERROR] {e}")
            raise
        
        # History storage
        self.history: List[dict] = []
        self.load_history()
        
        # Initialize performance metrics
        self.metrics = PerformanceMetrics(window_size=30)

        # Web dashboard metrics push (only used in full_duplex mode)
        self._last_metrics_emit_time: float = 0.0
        self._metrics_emit_interval_sec: float = 1.0
        self._metrics_emit_error_logged: bool = False
        self._metrics_emit_ok_logged: bool = False
        
        # Load English dictionary for word validation
        try:
            nltk.download('words', quiet=True)
            self.english_dictionary = set(w.lower() for w in words.words())
            print(f"[DICTIONARY] Loaded {len(self.english_dictionary)} words")
        except Exception as e:
            print(f"[DICTIONARY ERROR] {e}")
            self.english_dictionary = {"hello", "help", "home", "how", "have"}  # Fallback
        
        # Window resizing properties
        self.window_resizable = True
        self.current_window_size = None
        self.original_frame_size = None
        self.scale_factor = 1.0
        self.padding_color = (30, 30, 30)  # Dark gray background for padding
        
        # Get screen size for initial window sizing
        self.screen_width, self.screen_height = self.get_screen_size()
        
        # Pass metrics to WebSocket server for API access
        try:
            from websocket_server import set_global_metrics
            set_global_metrics(self.metrics)
        except Exception as e:
            print(f"[WARNING] Could not set metrics in WebSocket server: {e}")

    def _build_metrics_snapshot(self) -> dict:
        """Build a JSON-serializable metrics snapshot for the web dashboard."""
        summary = self.metrics.get_summary()

        gesture_stats = {}
        for gesture, stats in self.metrics.gesture_stats.items():
            gesture_stats[gesture] = {
                'attempts': stats.get('attempts', 0),
                'correct': stats.get('correct', 0),
                'accuracy': stats.get('accuracy', 0.0),
                'confidences': stats.get('confidences', [])
            }

        return {
            'fps': summary.get('fps', {'current': 0, 'min': 0, 'max': 0, 'target': 30}),
            'latency_ms': summary.get('latency_ms', {'current': 0, 'min': 0, 'max': 0, 'target': 33}),
            'confidence': summary.get('confidence', {'current': 0, 'min': 0, 'max': 0, 'target': 0.8}),
            'accuracy': summary.get('accuracy', {'overall': 0, 'correct_predictions': 0, 'total_predictions': 0}),
            'resources': summary.get('resources', {'memory_mb': 0, 'cpu_percent': 0}),
            'session': summary.get('session', {'duration_seconds': 0, 'total_frames': 0, 'total_gestures': 0}),
            'gesture_stats': gesture_stats
        }

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen resolution using tkinter or fallback method."""
        try:
            if TKINTER_AVAILABLE:
                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
                return screen_width, screen_height
        except:
            pass
        
        # Fallback: use common screen resolution
        return 1920, 1080

    def draw_ui_panel(self, img, x, y, w, h, title, accent_color=(0, 255, 200)):
        """Draws a semi-transparent panel with a glowing header line."""
        overlay = img.copy()
        # Dark translucent background
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 15, 15), -1)
        # Blend with original image
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        # Header line and text
        cv2.line(img, (x, y), (x + w, y), accent_color, 2)
        cv2.putText(img, title, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, accent_color, 2)

    def draw_scanning_brackets(self, img, x, y, w, h, color=(0, 255, 200)):
        """Draws tech-style corner brackets around the hand."""
        l = 25 # line length
        t = 2  # thickness
        # Top-Left
        cv2.line(img, (x, y), (x + l, y), color, t)
        cv2.line(img, (x, y), (x, y + l), color, t)
        # Top-Right
        cv2.line(img, (x + w, y), (x + w - l, y), color, t)
        cv2.line(img, (x + w, y), (x + w, y + l), color, t)
        # Bottom-Left
        cv2.line(img, (x, y + h), (x + l, y + h), color, t)
        cv2.line(img, (x, y + h), (x, y + h - l), color, t)
        # Bottom-Right
        cv2.line(img, (x + w, y + h), (x + w - l, y + h), color, t)
        cv2.line(img, (x + w, y + h), (x + w, y + h - l), color, t)
    
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
        # Kickstart dashboard with an immediate snapshot
        try:
            if hasattr(self, '_build_metrics_snapshot'):
                self.sio.emit('metrics_update', self._build_metrics_snapshot())
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
            
            # Stop any currently playing audio
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                pygame.mixer.quit()
                pygame.mixer.init()
            except Exception as e:
                print(f"[TTS MIXER RESET] {e}")
                try:
                    pygame.mixer.init()
                except:
                    pass
            
            # Create gTTS object
            tts = gTTS(text=text, lang="en", slow=False)
            
            # Save to temporary file with unique name
            temp_file = tempfile.mktemp(suffix=".mp3")
            tts.save(temp_file)
            
            # Play using pygame
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish with timeout
            max_wait_time = 30  # 30 seconds max wait
            wait_start = time.time()
            while pygame.mixer.music.get_busy() and (time.time() - wait_start) < max_wait_time:
                time.sleep(0.1)
            
            # Stop playback if still playing
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"[TTS ERROR] {e}")
            # Try to reset pygame mixer on error
            try:
                pygame.mixer.quit()
                pygame.mixer.init()
            except:
                pass

    # --- NON-BLOCKING _speak FUNCTION (Starts the thread) ---
    def _speak(self, text: str) -> None:
        """Speak text without blocking the main video loop"""
        if not text:
            return
        
        # Run the blocking speech call in a separate thread
        speech_thread = threading.Thread(target=self._speak_action, args=(text,), daemon=True)
        speech_thread.start()

    def load_history(self) -> None:
        """Load history from JSON file"""
        try:
            if HISTORY_FILE.exists():
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                print(f"[HISTORY] Loaded {len(self.history)} entries from history")
            else:
                self.history = []
                print("[HISTORY] No existing history file found, starting fresh")
        except Exception as e:
            print(f"[HISTORY] Error loading history: {e}")
            self.history = []

    def save_to_history(self, raw_text: str, corrected_text: str) -> None:
        """Save a new entry to history"""
        try:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'raw_text': raw_text,
                'corrected_text': corrected_text
            }
            self.history.append(entry)
            
            # Keep only last 100 entries to prevent file from growing too large
            if len(self.history) > 100:
                self.history = self.history[-100:]
            
            # Save to file
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            
            print(f"[HISTORY] Saved: '{corrected_text}'")
        except Exception as e:
            print(f"[HISTORY] Error saving to history: {e}")

    def show_history(self) -> None:
        """Display history in a separate window with delete option"""
        try:
            if not self.history:
                print("[HISTORY] No history to display")
                self._show_history_message("No history available", show_delete=False)
                return
            
            # Create a temporary window to display history
            history_text = "SIGN LANGUAGE RECOGNITION HISTORY\n"
            history_text += "=" * 40 + "\n\n"
            
            # Show last 20 entries in reverse chronological order
            recent_entries = self.history[-20:][::-1]
            for i, entry in enumerate(recent_entries, 1):
                timestamp = entry.get('timestamp', 'Unknown time')
                # Format timestamp to be more readable
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_time = timestamp
                
                corrected = entry.get('corrected_text', 'No text')
                raw = entry.get('raw_text', '')
                
                history_text += f"{i}. {formatted_time}\n"
                history_text += f"   Text: {corrected}\n"
                if raw != corrected:
                    history_text += f"   Corrected: {raw}\n"
                history_text += "\n"
            
            self._show_history_message(history_text, show_delete=True)
            print(f"[HISTORY] Displayed {len(recent_entries)} recent entries")
            
        except Exception as e:
            print(f"[HISTORY] Error displaying history: {e}")
            self._show_history_message(f"Error displaying history: {e}", show_delete=False)

    def _show_history_message(self, message: str, show_delete: bool = False) -> None:
        """Display a message in a temporary window with optional delete option"""
        try:
            # Create a blank image for the text display
            img_height = 700
            img_width = 700
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            # Add title
            cv2.putText(img, "HISTORY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Split message into lines and display
            lines = message.split('\n')
            y_offset = 80
            line_height = 25
            
            for line in lines:
                if y_offset > img_height - 60:  # Leave space for instructions
                    break  # Stop if we run out of space
                
                # Truncate long lines
                if len(line) > 80:
                    line = line[:77] + "..."
                
                # Color coding for different types of lines
                if line.startswith("SIGN LANGUAGE"):
                    color = (0, 255, 255)  # Yellow for title
                elif line.startswith("="):
                    color = (100, 100, 100)  # Gray for separators
                elif line.strip().endswith(":"):
                    color = (0, 255, 0)  # Green for timestamps
                elif line.startswith("   Text:"):
                    color = (255, 255, 255)  # White for corrected text
                elif line.startswith("   Raw:"):
                    color = (200, 200, 200)  # Light gray for raw text
                else:
                    color = (255, 255, 255)  # White for regular text
                
                cv2.putText(img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 1, cv2.LINE_AA)
                y_offset += line_height
            
            # Add instructions at bottom
            if show_delete and self.history:
                cv2.putText(img, "Press 'D' to delete all history, any other key to close", 
                           (20, img_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, "WARNING: This action cannot be undone!", 
                           (20, img_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "Press any key to close", (20, img_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display the window
            cv2.imshow("Sign Language History", img)
            
            # Wait for key press
            if show_delete and self.history:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('d') or key == ord('D'):
                    # Show confirmation dialog
                    if self._confirm_delete_history():
                        self.delete_all_history()
                        # Show confirmation message
                        self._show_delete_confirmation()
            else:
                cv2.waitKey(0)  # Wait for any key press
            
            cv2.destroyWindow("Sign Language History")
            
        except Exception as e:
            print(f"[HISTORY] Error showing history window: {e}")

    def _show_alphabet_signs(self) -> None:
        """Display the alphabet signs image"""
        try:
            # Load the signs image
            img_path = Path(__file__).parent / 'signs._image.png'
            
            # Try alternative file names if not found
            if not img_path.exists():
                img_path = Path(__file__).parent / 'signs_image.png.jpg'
            if not img_path.exists():
                img_path = Path(__file__).parent / 'signs_image.png'
            if not img_path.exists():
                img_path = Path(__file__).parent / 'signs_image.jpg'
            
            img = cv2.imread(str(img_path))
            
            if img is None:
                print("[ALPHABET] Error: Could not load signs_image.png.jpg")
                print(f"[ALPHABET] Looking for file at: {img_path}")
                self._show_message("Image Not Found", "Please place signs_image.png.jpg in the project folder")
                return
                
            # Resize to fit screen if needed
            height, width = img.shape[:2]
            max_height = 800
            if height > max_height:
                scale = max_height / height
                img = cv2.resize(img, (int(width * scale), max_height))
            
            # Show the image
            cv2.imshow("Alphabet Signs", img)
            cv2.waitKey(0)
            cv2.destroyWindow("Alphabet Signs")
            
        except Exception as e:
            print(f"[ALPHABET] Error showing alphabet signs: {e}")
            self._show_message("Error", f"Could not display alphabet signs: {e}")

    def _show_message(self, title: str, message: str) -> None:
        """Show a simple message window"""
        try:
            img = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(img, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Wrap message if too long
            words = message.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 60:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            if current_line:
                lines.append(current_line)
            
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                cv2.putText(img, line, (20, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow("Message", img)
            cv2.waitKey(2000)
            cv2.destroyWindow("Message")
        except Exception as e:
            print(f"[MESSAGE] Error: {e}")

    def get_word_suggestions(self) -> List[str]:
        """Finds top 3 dictionary words matching the current buffer's last word."""
        # Get the last word currently being typed in the buffer
        current_text = "".join(self.buffer)
        if not current_text or current_text.endswith(" "):
            return []
        
        current_word = current_text.split()[-1].lower()
        if len(current_word) < 2:  # Only suggest after 2 letters
            return []

        # Filter dictionary for words starting with current_word
        matches = [w for w in self.english_dictionary if w.startswith(current_word)]
        # Sort by length so shorter, more common words appear first
        matches.sort(key=len)
        return matches[:3]

    def draw_suggestion_hud(self, frame, suggestions):
        """Draws a high-tech selection bar for word suggestions."""
        if not suggestions:
            return
        
        h, w = frame.shape[:2]
        panel_x, panel_y = 20, h - 250  # Positioned above the Correction panel
        panel_w, panel_h = 450, 45
        
        # Draw glassmorphic background
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 255), 1)

        # Render suggestions as 'Hotkeys'
        cv2.putText(frame, "SUGGEST [1-3]:", (panel_x + 10, panel_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        for i, word in enumerate(suggestions):
            text = f"({i+1}) {word.upper()}"
            cv2.putText(frame, text, (panel_x + 130 + (i * 100), panel_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _confirm_delete_history(self) -> bool:
        """Show confirmation dialog for deleting history"""
        try:
            # Create confirmation dialog
            img_height = 200
            img_width = 500
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            # Add warning message
            cv2.putText(img, "DELETE ALL HISTORY?", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.putText(img, "This will permanently delete all", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.putText(img, f"{len(self.history)} history entries.", (50, 115), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.putText(img, "Press 'Y' to confirm, any other key to cancel", (50, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.imshow("Confirm Delete History", img)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow("Confirm Delete History")
            
            return key == ord('y') or key == ord('Y')
            
        except Exception as e:
            print(f"[HISTORY] Error showing confirmation dialog: {e}")
            return False

    def delete_all_history(self) -> None:
        """Delete all history entries"""
        try:
            self.history = []
            
            # Delete the history file if it exists
            if HISTORY_FILE.exists():
                HISTORY_FILE.unlink()
            
            print(f"[HISTORY] Deleted all history entries")
            
        except Exception as e:
            print(f"[HISTORY] Error deleting history: {e}")

    def _show_delete_confirmation(self) -> None:
        """Show confirmation message after successful deletion"""
        try:
            # Create confirmation message
            img_height = 150
            img_width = 400
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            cv2.putText(img, "HISTORY DELETED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(img, "All history entries have been", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.putText(img, "permanently deleted.", (50, 115), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow("History Deleted", img)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyWindow("History Deleted")
            
        except Exception as e:
            print(f"[HISTORY] Error showing delete confirmation: {e}")

    def _predict_label(self, feats: np.ndarray) -> Tuple[str, float]:
        # Start processing timer
        self.metrics.start_processing()
        
        label = None
        conf = 0.0
        
        # Use multi-language recognizer if available, fall back to legacy model
        try:
            if self.ml_recognizer and hasattr(self.ml_recognizer, 'models') and self.ml_recognizer.models:
                label, conf = self.ml_recognizer.predict(feats)
            elif self.model:
                pred_idx = self.model.predict([feats])[0]
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba([feats])[0]
                    conf = float(np.max(probs))
                else:
                    conf = 1.0
                label = self.labels[pred_idx]
            else:
                raise RuntimeError("No model available for prediction")
        except Exception as e:
            print(f"[PREDICTION ERROR] {e}")
            return None, 0.0
        
        # Try ISL dual hands model if available and feature size is 126
        if self.isl_model and len(feats) == 126:
            try:
                isl_prediction = self.isl_model.predict([feats])[0]
                isl_confidence = self.isl_model.predict_proba([feats]).max()
                # Use ISL prediction if confidence is higher
                if isl_confidence > conf:
                    label = isl_prediction
                    conf = isl_confidence
            except Exception as e:
                pass  # Silently fail if ISL prediction doesn't work
        
        # End processing timer
        self.metrics.end_processing()
        
        # Record prediction for metrics
        if label:
            self.metrics.record_prediction(label, conf, is_correct=True)
        
        return label, conf

    def _toggle_language(self) -> None:
        """Toggle between ASL and ISL languages"""
        try:
            current_lang = self.language_manager.get_current_language()
            available_langs = self.language_manager.SUPPORTED_LANGUAGES
            
            # Get next language in the list
            current_idx = available_langs.index(current_lang)
            next_lang = available_langs[(current_idx + 1) % len(available_langs)]
            
            self._switch_language(next_lang)
        except Exception as e:
            print(f"[LANGUAGE ERROR] Could not toggle language: {e}")

    def _switch_language(self, language: str) -> None:
        """Switch to a specific language (ASL or ISL)"""
        try:
            if language not in self.language_manager.SUPPORTED_LANGUAGES:
                print(f"[LANGUAGE] Unsupported language: {language}")
                return
            
            # Check if model is loaded for this language
            if language not in self.ml_recognizer.models:
                print(f"[LANGUAGE] No model found for {language}. Available models: {list(self.ml_recognizer.models.keys())}")
                return
            
            # Switch language in the manager
            self.language_manager.set_current_language(language)
            
            # Update labels from the new language's model
            model_info = self.ml_recognizer.get_model_info(language)
            self.labels = model_info.get("labels", [])
            
            print(f"[LANGUAGE] ✓ Switched to {language} | Available symbols: {', '.join(self.labels[:10])}...")
            
        except Exception as e:
            print(f"[LANGUAGE ERROR] Could not switch to {language}: {e}")

    def _show_welcome_window(self) -> None:
        """Display a welcome window with image for 2 seconds before starting the camera"""
        try:
            # Try to load the welcome image
            welcome_image_path = Path("C:\\Users\\samir\\OneDrive\\Desktop\\3.0\\welcome.png")
            
            if welcome_image_path.exists():
                # Load and display the welcome image for 2 seconds
                welcome_img = cv2.imread(str(welcome_image_path))
                if welcome_img is not None:
                    h, w = welcome_img.shape[:2]
                    
                    # Create resizable window for welcome screen
                    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
                    
                    # Calculate optimal window size for better clarity
                    # Use a smaller percentage to ensure image is clear and not stretched
                    max_w = int(self.screen_width * 0.8)  # Reduced to 0.6 for better clarity
                    max_h = int(self.screen_height * 0.8)  # Reduced to 0.6 for better clarity
                    
                    # Check if image needs to be resized
                    if w > max_w or h > max_h:
                        # Calculate scale factor to fit while maintaining aspect ratio
                        scale_w = max_w / w
                        scale_h = max_h / h
                        scale = min(scale_w, scale_h)
                        
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        
                        # Resize welcome image with high-quality interpolation
                        welcome_img = cv2.resize(welcome_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        
                        # Update dimensions for centering
                        w, h = new_w, new_h
                    
                    cv2.resizeWindow(WINDOW_TITLE, w, h)
                    
                    # Center on screen
                    screen_x = (self.screen_width - w) // 2
                    screen_y = (self.screen_height - h) // 2
                    cv2.moveWindow(WINDOW_TITLE, screen_x, screen_y)
                    
                    cv2.imshow(WINDOW_TITLE, welcome_img)
                    cv2.waitKey(2000)  # Display for 3 seconds
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
            
            # Create resizable window for welcome screen
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            
            # Check if window is maximized and adjust size accordingly
            try:
                # Get current window state
                window_rect = cv2.getWindowImageRect(WINDOW_TITLE)
                current_width = window_rect[2]
                current_height = window_rect[3]
                
                # Check if window is maximized (close to screen size)
                if (abs(current_width - self.screen_width) < 50 and 
                    abs(current_height - self.screen_height) < 50):
                    # Window is maximized, resize welcome frame to fit screen
                    max_w = int(self.screen_width * 0.7)  # Reduced from 0.95 to 0.7
                    max_h = int(self.screen_height * 0.7)  # Reduced from 0.95 to 0.7
                    
                    # Calculate scale factor to fit while maintaining aspect ratio
                    scale_w = max_w / w
                    scale_h = max_h / h
                    scale = min(scale_w, scale_h)
                    
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Resize frame
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    h, w = frame.shape[:2]
                    
                    # Center the resized frame
                    screen_x = (self.screen_width - new_w) // 2
                    screen_y = (self.screen_height - new_h) // 2
                    cv2.moveWindow(WINDOW_TITLE, screen_x, screen_y)
                else:
                    # Window not maximized, use original size
                    cv2.resizeWindow(WINDOW_TITLE, w, h)
                    # Center on screen
                    screen_x = (self.screen_width - w) // 2
                    screen_y = (self.screen_height - h) // 2
                    cv2.moveWindow(WINDOW_TITLE, screen_x, screen_y)
            except:
                # Fallback to original size if window detection fails
                cv2.resizeWindow(WINDOW_TITLE, w, h)
                screen_x = (self.screen_width - w) // 2
                screen_y = (self.screen_height - h) // 2
                cv2.moveWindow(WINDOW_TITLE, screen_x, screen_y)
            
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
            cv2.putText(frame, 'Make hand gestures to recognize signs', (int(w*0.1), int(h*0.55)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Controls section
            cv2.putText(frame, 'CONTROLS:', (int(w*0.1), int(h*0.62)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(frame, "'s' - Space | 'h' - History | 'a' - Alphabet Signs", (int(w*0.1), int(h*0.68)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            
            cv2.putText(frame, "Space/Enter - Speak", (int(w*0.1), int(h*0.74)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            
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

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 2
            rec_text = f"Recognized: {spoken_text}"
            (tw, th), bl = cv2.getTextSize(rec_text, font, scale, thickness)
            x0, y0 = 15, 85
            x1, y1 = x0 + tw + 20, y0 + th + 20
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
            cv2.putText(frame, rec_text, (x0 + 10, y0 + th + 5), font, scale, (255, 255, 0), thickness)

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

        print("[START] Gesture recognition system starting...")

        # Mouse callback function for close button, speak button, history button, and performance button
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if click is within close button rectangle
                if hasattr(self, 'close_button_rect'):
                    x1, y1, x2, y2 = self.close_button_rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.corrected_display_text = ""
                        self.display_correction_until = 0
                        print("[UI] AI Correction window closed by mouse click")
                
                # Check if click is within speak button rectangle
                if hasattr(self, 'speak_button_rect'):
                    x1, y1, x2, y2 = self.speak_button_rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        # Speak the recognized text
                        text = "".join(self.buffer).strip()
                        if text:
                            # Apply grammar correction to the recognized text
                            corrected_text = correct_grammar(text)
                            print(f"[Grammar] Raw: {text}")
                            print(f"[Grammar] Corrected: {corrected_text}")
                            
                            # Store corrected text for display
                            self.corrected_display_text = corrected_text
                            self.display_correction_until = time.time() + 10.0  # Display for 10 seconds
                            
                            # Save to history
                            self.save_to_history(text, corrected_text)
                            
                            # Send full sentence to web via WebSocket (if connected)
                            if self.full_duplex and self.sio and self.sio.connected:
                                try:
                                    emotion_data = {'emotion': self.current_emotion, 'confidence': self.emotion_confidence} if self.current_emotion != "neutral" else None
                                    self.sio.emit('sign_recognized', {
                                        'text': corrected_text,
                                        'emotion': emotion_data
                                    })
                                    print(f"[WebSocket] Sent full sentence: '{corrected_text}' with emotion: {emotion_data}")
                                except Exception as e:
                                    print(f"[WebSocket] Could not send full sentence: {e}")
                            
                            # Speak the corrected text
                            self._speak(corrected_text)
                            print("[UI] Speak button clicked - speaking recognized text")
                        else:
                            print("[UI] No text to speak")
                
                # Check if click is within history button rectangle
                if hasattr(self, 'history_button_rect'):
                    x1, y1, x2, y2 = self.history_button_rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        # Show history
                        self.show_history()
                        print("[UI] History button clicked - showing history")
                
                # Check if click is within performance button rectangle
                if hasattr(self, 'performance_button_rect'):
                    x1, y1, x2, y2 = self.performance_button_rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        # Open performance dashboard in browser
                        import webbrowser
                        try:
                            # Open the performance dashboard URL with correct .html extension
                            dashboard_url = "http://localhost:5000/performance.html"
                            webbrowser.open(dashboard_url)
                            print(f"[UI] Performance button clicked - opening dashboard: {dashboard_url}")
                        except Exception as e:
                            print(f"[UI] Error opening performance dashboard: {e}")
                            # Fallback: try opening localhost:8000 with .html if 5000 doesn't work
                            try:
                                fallback_url = "http://localhost:8000/performance.html"
                                webbrowser.open(fallback_url)
                                print(f"[UI] Opening fallback dashboard: {fallback_url}")
                            except Exception as e2:
                                print(f"[UI] Could not open any performance dashboard: {e2}")
                
                # Check if click is within voice button rectangle
                if hasattr(self, 'voice_button_rect'):
                    x1, y1, x2, y2 = self.voice_button_rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        # Set flag to trigger voice recognition in main loop
                        self.voice_input_requested = True
                        print("[UI] Voice button clicked - activating voice recognition")
                
                # Check if click is within alphabet button rectangle
                if hasattr(self, 'alphabet_button_rect'):
                    x1, y1, x2, y2 = self.alphabet_button_rect
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        # Show alphabet signs
                        self._show_alphabet_signs()
                        print("[UI] Alphabet button clicked - showing alphabet signs")

        # Show welcome window before starting camera
        self._show_welcome_window()

        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            raise RuntimeError("Cannot access webcam")
        
        # Set window properties for resizing
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_TITLE, mouse_callback)
        
        # Set initial window size based on screen resolution
        window_width = int(self.screen_width * 0.98)  # 98% of screen width (further increased from right)
        window_height = int(self.screen_height * 0.95)  # 95% of screen height (increased from bottom)
        
        # Ensure minimum window size
        window_width = max(window_width, 800)
        window_height = max(window_height, 600)
        
        cv2.resizeWindow(WINDOW_TITLE, window_width, window_height)
        
        # Position window more to the right side and slightly lower
        screen_x = int((self.screen_width - window_width) * 0.7)  # 10% from left (shifted further right)
        screen_y = int((self.screen_height - window_height) * 0.3)  # 30% from top (adjusted for larger window)
        cv2.moveWindow(WINDOW_TITLE, screen_x, screen_y)

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lowered from default 0.5
            min_tracking_confidence=0.3    # Lowered from default 0.5
        )
        extractor = HandFeatureExtractor()

        # Load the trained ISL model
        try:
            isl_model = joblib.load('models/sign_knn_isl_dual_hands.joblib')
            print("✅ ISL model loaded successfully")
        except:
            isl_model = None
            print("⚠️ ISL model not found")

        frame_count = 0
        while True:
            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames (approx 1 second at 30 FPS)
                print(f"[LOOP] Frame {frame_count}, Buffer: {''.join(self.buffer)}, Stable: {self.stable_count}")
            
            # Start frame timing
            self.metrics.start_frame()
            
            ok, frame = cap.read()
            if not ok:
                break
            if self.flip:
                frame = cv2.flip(frame, 1)

            # Draw header and status UI
            h, w = frame.shape[:2]
            
            # Check window size and calculate scale factor
            try:
                window_rect = cv2.getWindowImageRect(WINDOW_TITLE)
                window_width = window_rect[2]
                window_height = window_rect[3]
                
                # Store original frame size for scaling
                if self.original_frame_size is None:
                    self.original_frame_size = (w, h)
                
                # Calculate scale factor to fit frame in window with padding
                padding = 100  # Total padding (50px on each side)
                max_frame_width = window_width - padding
                max_frame_height = window_height - padding
                
                # Calculate scale factor maintaining aspect ratio
                scale_w = max_frame_width / w
                scale_h = max_frame_height / h
                self.scale_factor = min(scale_w, scale_h, 2.0)  # Cap at 2x for performance
                
            except:
                # Fallback to scale factor 1.0 if window size detection fails
                window_width = w
                window_height = h
                self.scale_factor = 1.0
            
            # Scale frame if needed
            if self.scale_factor > 1.0:
                new_w = int(w * self.scale_factor)
                new_h = int(h * self.scale_factor)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                h, w = frame.shape[:2]
            
            # Get current window size for centering
            try:
                window_rect = cv2.getWindowImageRect(WINDOW_TITLE)
                window_width = window_rect[2]
                window_height = window_rect[3]
            except:
                window_width = w
                window_height = h
            
            # Create frame with camera positioned to the right side
            if window_width > w or window_height > h:
                # Calculate positioning offsets - move camera to right side and increase height
                x_offset = window_width - w - 50  # Position 50px from right edge
                y_offset = 20  # Position 20px from top (reduced from centering to increase height)
                
                # Create padded frame with background color
                padded_frame = np.full((window_height, window_width, 3), self.padding_color, dtype=np.uint8)
                
                # Place the original frame in the center
                padded_frame[y_offset:y_offset + h, x_offset:x_offset + w] = frame
                
                # Use padded frame for drawing UI
                frame = padded_frame
            
            # Show screen size (width x height)
            screen_text = f'{w}x{h}'
            
            # Commented out the gray header background
            # cv2.rectangle(frame, (0,0), (w, 100), (50,50,50), -1)
            # Header background
            
            # Prediction and accuracy display (use last_confidence as realtime accuracy indicator)
            # We'll render accuracy below the 'Pred' line and above the 'Text' area
            conf_pct = self.last_confidence * 100.0
            acc_text = f'Accuracy: {conf_pct:0.2f}%'

            # --- REPLACEMENT DRAWING LOGIC ---
            h, w, _ = frame.shape
            MAIN_NEON = (210, 255, 0) # Cyan-Teal color from your reference image

            # Process hand landmarks with MediaPipe FIRST
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                res = hands.process(rgb)
            except Exception as e:
                print(f"[MediaPipe] Error processing frame: {e}")
                res = None  # Continue with None result

            # 1. Prediction Hero Box (Top Right)
            self.draw_ui_panel(frame, w - 320, 30, 200, 120, "GESTURE RECOGNITION", MAIN_NEON)
            pred_char = self.current_letter.upper() if self.current_letter else "--"
            cv2.putText(frame, pred_char, (w - 240, 105), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3)

            # 1B. Language Indicator Panel (Top Right, below Gesture Recognition)
            current_language = self.language_manager.get_current_language()
            lang_color = (0, 255, 0) if current_language == "ASL" else (0, 165, 255) if current_language == "ISL" else (100, 100, 100)
            self.draw_ui_panel(frame, w - 320, 160, 200, 50, f"LANGUAGE: {current_language}", lang_color)
            cv2.putText(frame, "Press L to toggle | 1:ASL 2:ISL", (w - 315, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # 2. Live Accuracy Score Panel (Left Side)
            # Calculate live accuracy based on recent predictions with more realistic scoring
            recent_window = 10  # Last 10 predictions
            if hasattr(self, 'prediction_history') and len(self.prediction_history) > 0:
                recent_predictions = self.prediction_history[-recent_window:]
                if len(recent_predictions) > 0:
                    # Calculate weighted accuracy based on confidence levels
                    total_score = 0
                    for pred in recent_predictions:
                        confidence = pred.get('confidence', 0)
                        # Use confidence as direct accuracy score (more realistic)
                        total_score += confidence * 100
                    
                    accuracy_val = total_score / len(recent_predictions)
                else:
                    accuracy_val = self.last_confidence * 100
            else:
                # Use current confidence as live accuracy
                accuracy_val = self.last_confidence * 100
            
            self.draw_ui_panel(frame, 20, 30, 290, 100, "LIVE  ACCURACY", MAIN_NEON)
            cv2.putText(frame, f"ACCURACY: {accuracy_val:.2f}%", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            # Draw accuracy bar
            bar_width = int(240 * (accuracy_val / 100))
            cv2.rectangle(frame, (35, 95), (275, 110), (50, 50, 50), -1) # Background
            cv2.rectangle(frame, (35, 95), (35 + bar_width, 110), MAIN_NEON, -1) # Fill

            # 3. Message Buffer Panel (Left Side)
            # Calculate maximum width for message panel to avoid video overlap
            video_start_x = window_width - w - 50  # Video starts here
            max_panel_width = video_start_x - 40  # Leave 40px margin from video
            panel_width = min(650, max_panel_width)  # Use smaller of 550 or calculated max
            
            self.draw_ui_panel(frame, 20, 150, panel_width, 120, "CURRENT MESSAGE", (0, 200, 255))
            msg_str = "".join(self.buffer)
            
            # Calculate max characters per line based on panel width
            # Approx 8 pixels per character for the font size used
            max_chars_per_line = max(20, (panel_width - 70) // 8)  # Leave margin for panel border
            
            # Text wrapping for long messages
            lines = []
            current_line = ""
            
            for char in msg_str[-100:]:  # Show last 100 characters with wrapping
                if len(current_line) >= max_chars_per_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line += char
            
            if current_line:
                lines.append(current_line)
            
            # Display wrapped text
            y_start = 195  # Moved down to avoid title overlap
            for i, line in enumerate(lines[-3:]):  # Show last 3 lines
                if i == 0:
                    display_text = f"> {line}"
                else:
                    display_text = f"  {line}"  # Indent continuation lines
                cv2.putText(frame, display_text, (35, y_start + (i * 25)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 4. Emotion Indicator (Bottom Left - Moved down to avoid overlap)
            emo_color = (0, 255, 0) if self.current_emotion == "happy" else (255, 255, 255)
            cv2.rectangle(frame, (20, h - 120), (220, h - 70), (10, 10, 10), -1)
            cv2.putText(frame, f"MOOD: {self.current_emotion.upper()}", (40, h - 88), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, emo_color, 2)

            # 5. Controls Panel (Positioned below Emotion section)
            controls_y_start = h - 320  # Fixed position from bottom to ensure visibility
            self.draw_ui_panel(frame, 20, controls_y_start, 300, 160, "SYSTEM CONTROLS", (255, 200, 0))
            
            controls_text = [
                "ESC - Exit System",
                "Space/Enter - Speak Sentence",
                "Backspace - Delete",
                "S - Add Space",
                "H - View History"
            ]
            
            for i, control in enumerate(controls_text):
                cv2.putText(frame, control, (35, controls_y_start + 45 + (i * 25)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 6. Hand Targeting Brackets (Dynamic)
            if res and res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    # Calculate bounding box of hand
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
                    
                    # Draw tech brackets instead of just lines
                    self.draw_scanning_brackets(frame, x_min-30, y_min-30, (x_max-x_min)+60, (y_max-y_min)+60, MAIN_NEON)
                    
                    # Optional: draw landmarks with the same neon color
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=MAIN_NEON, thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))

            # 6. Emotion Indicator (Bottom Left - Moved down to avoid overlap)
            emo_color = (0, 255, 0) if self.current_emotion == "happy" else (255, 255, 255)
            cv2.rectangle(frame, (20, h - 120), (220, h - 70), (10, 10, 10), -1)
            cv2.putText(frame, f"MOOD: {self.current_emotion.upper()}", (40, h - 88), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, emo_color, 2)

            # 7. AI CORRECTION (Bottom Left)
            if self.corrected_display_text:
                self.draw_ui_panel(frame, 20, h - 180, 450, 70, "AI SMART CORRECTION", (0, 255, 100))
                # Position text within the panel bounds (panel starts at h-180, height 70)
                text_y = h - 180 + 45  # 45px from panel top (below title)
                cv2.putText(frame, self.corrected_display_text[:40], (35, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Add close button (X) in upper right corner
                close_button_x = 20 + 450 - 30  # 30px from right edge
                close_button_y = h - 180 + 10   # 10px from top
                close_button_size = 20
                
                # Draw close button background (red circle)
                cv2.circle(frame, (close_button_x, close_button_y), close_button_size, (0, 0, 255), -1)
                cv2.circle(frame, (close_button_x, close_button_y), close_button_size, (255, 255, 255), 2)
                
                # Draw X symbol
                cv2.line(frame, (close_button_x - 8, close_button_y - 8), (close_button_x + 8, close_button_y + 8), (255, 255, 255), 3)
                cv2.line(frame, (close_button_x + 8, close_button_y - 8), (close_button_x - 8, close_button_y + 8), (255, 255, 255), 3)
                
                # Store close button position for click detection
                self.close_button_rect = (close_button_x - close_button_size, close_button_y - close_button_size, 
                                         close_button_x + close_button_size, close_button_y + close_button_size)

            # 8. SPEAK BUTTON (Bottom Right Corner of Video Feed)
            # Position relative to video feed area (right side of window)
            speak_button_x = w - 100  # Position near bottom right
            speak_button_y = h - 80
            speak_button_size = 25
            cv2.circle(frame, (speak_button_x, speak_button_y), speak_button_size, (0, 255, 0), -1)
            cv2.circle(frame, (speak_button_x, speak_button_y), speak_button_size, (255, 255, 255), 3)
            
            # Draw sound/speaker icon (simplified)
            # Speaker cone
            cone_points = np.array([
                [speak_button_x - 8, speak_button_y - 8],
                [speak_button_x + 2, speak_button_y - 12],
                [speak_button_x + 2, speak_button_y + 12],
                [speak_button_x - 8, speak_button_y + 8]
            ], np.int32)
            cv2.fillPoly(frame, [cone_points], (255, 255, 255))
            # Speaker body
            cv2.rectangle(frame, (speak_button_x + 2, speak_button_y - 6), (speak_button_x + 8, speak_button_y + 6), (255, 255, 255), -1)
            # Sound waves
            cv2.line(frame, (speak_button_x + 10, speak_button_y - 4), (speak_button_x + 14, speak_button_y - 6), (255, 255, 255), 2)
            cv2.line(frame, (speak_button_x + 10, speak_button_y), (speak_button_x + 15, speak_button_y), (255, 255, 255), 2)
            cv2.line(frame, (speak_button_x + 10, speak_button_y + 4), (speak_button_x + 14, speak_button_y + 6), (255, 255, 255), 2)
            
            # Add "SPEAK" text below button
            cv2.putText(frame, "SPEAK", (speak_button_x - 20, speak_button_y + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
            
            # Store speak button position for click detection
            self.speak_button_rect = (speak_button_x - speak_button_size, speak_button_y - speak_button_size, 
                                      speak_button_x + speak_button_size, speak_button_y + speak_button_size)

            # 9. HISTORY BUTTON (Rectangular, next to speak button)
            history_button_x = speak_button_x - 120  # Increased from 80 to 120px for more space
            history_button_y = speak_button_y - 15  # Center vertically with speak button
            history_button_width = 50  # Reduced from 60 to 50
            history_button_height = 30  # Reduced from 40 to 30
            
            # Draw history button background (blue rectangle)
            cv2.rectangle(frame, (history_button_x, history_button_y), 
                         (history_button_x + history_button_width, history_button_y + history_button_height), 
                         (0, 100, 255), -1)
            cv2.rectangle(frame, (history_button_x, history_button_y), 
                         (history_button_x + history_button_width, history_button_y + history_button_height), 
                         (255, 255, 255), 2)
            
            # Draw history icon (simplified book icon, smaller)
            # Book icon
            cv2.rectangle(frame, (history_button_x + 12, history_button_y + 8), 
                         (history_button_x + 38, history_button_y + 22), (255, 255, 255), -1)
            cv2.line(frame, (history_button_x + 25, history_button_y + 8), 
                    (history_button_x + 25, history_button_y + 22), (0, 100, 255), 2)
            
            # Add "HISTORY" text below button
            cv2.putText(frame, "HISTORY", (history_button_x - 8, history_button_y + 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Store history button position for click detection
            self.history_button_rect = (history_button_x, history_button_y, 
                                       history_button_x + history_button_width, history_button_y + history_button_height)

            # 10. PERFORMANCE BUTTON (Rectangular, next to history button)
            performance_button_x = history_button_x - 80  # 80px to the left of history button
            performance_button_y = speak_button_y - 15  # Center vertically with other buttons
            performance_button_width = 50
            performance_button_height = 30
            
            # Draw performance button background (orange rectangle)
            cv2.rectangle(frame, (performance_button_x, performance_button_y), 
                         (performance_button_x + performance_button_width, performance_button_y + performance_button_height), 
                         (0, 165, 255), -1)  # Orange color
            cv2.rectangle(frame, (performance_button_x, performance_button_y), 
                         (performance_button_x + performance_button_width, performance_button_y + performance_button_height), 
                         (255, 255, 255), 2)
            
            # Draw performance icon (simplified chart icon)
            # Chart bars
            cv2.rectangle(frame, (performance_button_x + 10, performance_button_y + 20), 
                         (performance_button_x + 18, performance_button_y + 10), (255, 255, 255), -1)
            cv2.rectangle(frame, (performance_button_x + 20, performance_button_y + 20), 
                         (performance_button_x + 28, performance_button_y + 15), (255, 255, 255), -1)
            cv2.rectangle(frame, (performance_button_x + 30, performance_button_y + 20), 
                         (performance_button_x + 38, performance_button_y + 5), (255, 255, 255), -1)
            
            # Add "PERF" text below button
            cv2.putText(frame, "PERF", (performance_button_x - 5, performance_button_y + 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Store performance button position for click detection
            self.performance_button_rect = (performance_button_x, performance_button_y, 
                                           performance_button_x + performance_button_width, performance_button_y + performance_button_height)

            # 11. VOICE INPUT BUTTON (Left of Alphabet Button)
            # Position to the left of alphabet button in the same line
            voice_button_x = performance_button_x - 160  # 160px to the left of performance button
            voice_button_y = performance_button_y  # Same vertical position as other buttons
            voice_button_width = 50
            voice_button_height = 30
            
            # Draw voice button background (red rectangle)
            cv2.rectangle(frame, (voice_button_x, voice_button_y), 
                         (voice_button_x + voice_button_width, voice_button_y + voice_button_height), 
                         (0, 0, 255), -1)  # Red color
            cv2.rectangle(frame, (voice_button_x, voice_button_y), 
                         (voice_button_x + voice_button_width, voice_button_y + voice_button_height), 
                         (255, 255, 255), 2)
            
            # Draw microphone icon
            # Mic body
            cv2.rectangle(frame, (voice_button_x + 22, voice_button_y + 8), (voice_button_x + 28, voice_button_y + 20), (255, 255, 255), -1)
            # Mic head
            cv2.ellipse(frame, (voice_button_x + 25, voice_button_y + 5), (4, 3), 0, 0, 360, (255, 255, 255), -1)
            # Mic stand
            cv2.line(frame, (voice_button_x + 25, voice_button_y + 20), (voice_button_x + 25, voice_button_y + 25), (255, 255, 255), 2)
            cv2.line(frame, (voice_button_x + 20, voice_button_y + 25), (voice_button_x + 30, voice_button_y + 25), (255, 255, 255), 2)
            
            # Add "VOICE" text below button
            cv2.putText(frame, "VOICE", (voice_button_x + 5, voice_button_y + 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Store voice button position for click detection
            self.voice_button_rect = (voice_button_x, voice_button_y, 
                                     voice_button_x + voice_button_width, voice_button_y + voice_button_height)

            # 12. ALPHABET BUTTON (Left of Performance Button)
            # Position to the left of performance button in the same line
            alphabet_button_x = performance_button_x - 80  # 80px to the left of performance button
            alphabet_button_y = performance_button_y  # Same vertical position as other buttons
            alphabet_button_width = 50
            alphabet_button_height = 30
            
            # Draw alphabet button background (purple rectangle)
            cv2.rectangle(frame, (alphabet_button_x, alphabet_button_y), 
                         (alphabet_button_x + alphabet_button_width, alphabet_button_y + alphabet_button_height), 
                         (128, 0, 255), -1)  # Purple color
            cv2.rectangle(frame, (alphabet_button_x, alphabet_button_y), 
                         (alphabet_button_x + alphabet_button_width, alphabet_button_y + alphabet_button_height), 
                         (255, 255, 255), 2)
            
            # Draw alphabet icon (simplified ABC icon)
            # ABC text as icon
            cv2.putText(frame, "ABC", (alphabet_button_x + 8, alphabet_button_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add "ALPHABET" text below button
            cv2.putText(frame, "ALPHABET", (alphabet_button_x - 5, alphabet_button_y + 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Store alphabet button position for click detection
            self.alphabet_button_rect = (alphabet_button_x, alphabet_button_y, 
                                        alphabet_button_x + alphabet_button_width, alphabet_button_y + alphabet_button_height)

            # Process emotion detection if available
            if self.emotion_detector:
                try:
                    self.current_emotion, self.emotion_confidence = self.emotion_detector.detect_emotion(frame)
                    # Position emotion info below all text elements
                    emotion_start_y = 300  # Fixed position since we removed old text layout
                    frame = self.emotion_detector.draw_emotion_info(frame, self.current_emotion, self.emotion_confidence, emotion_start_y)
                except Exception as e:
                    print(f"[EMOTION] Error detecting emotion: {e}")

            # Process hand landmarks if any
            if res.multi_hand_landmarks:
                print(f"[HAND DETECTED] Hands detected: {len(res.multi_hand_landmarks)}")
                # Draw hand landmarks on frame (adjust for centered frame)
                for hand_landmarks in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                
                pred_label = None
                conf = 0.0
                try:
                    feats = extractor.from_mediapipe(res.multi_hand_landmarks[0])
                    print(f"[FEATURES] Shape: {feats.shape}, Range: [{feats.min():.3f}, {feats.max():.3f}]")
                    pred_label, conf = self._predict_label(feats)
                    print(f"[PREDICTION] Label: {pred_label}, Conf: {conf:.3f}")
                    # store last confidence for UI
                    self.last_confidence = conf
                except Exception as e:
                    print(f"[ERROR] Feature extraction/prediction: {e}")
                    import traceback
                    traceback.print_exc()
                    pred_label = None
                    conf = 0.0
                
                # Use language-specific confidence thresholds
                current_lang = self.language_manager.get_current_language()
                min_conf_threshold = MIN_CONFIDENCE_ISL if current_lang == "ISL" else MIN_CONFIDENCE
                
                if pred_label and conf < min_conf_threshold:
                    print(f"[FILTERED] {pred_label} confidence {conf:.3f} below threshold {min_conf_threshold}")
                    pred_label = None
                elif pred_label:
                    print(f"[ACCEPTED] {pred_label} (conf: {conf:.3f}, threshold: {min_conf_threshold})")
            else:
                print(f"[NO HANDS] No hands detected")
                pred_label = None

            if pred_label:
                self.pred_window.append(pred_label)
                cnt = Counter(self.pred_window)
                maj_label, maj_count = cnt.most_common(1)[0]
                if maj_count >= max(3, PRED_SMOOTHING // 2):
                    self.stable_count += 1
                    
                    # First time detecting this letter
                    if self.stable_count == DEBOUNCE_FRAMES:
                        # Initialize hold tracking
                        self.letter_hold_start_time = time.time()
                        self.letter_added_once = False
                        self.double_letter_added = False
                        self.last_committed = maj_label
                        self.current_letter = maj_label
                    
                    # Add letter only once on first detection
                    if self.stable_count == DEBOUNCE_FRAMES and not self.letter_added_once:
                        self.buffer.append(maj_label)
                        self.letter_added_once = True
                        self._last_buffer_activity = time.time()
                        print(f"[SIGN] Added letter: {maj_label}")
                        
                        # Track prediction for live accuracy
                        if not hasattr(self, 'prediction_history'):
                            self.prediction_history = []
                        
                        # Store prediction with confidence score
                        self.prediction_history.append({
                            'letter': maj_label,
                            'confidence': self.last_confidence,
                            'timestamp': time.time()
                        })
                        
                        # Keep only last 50 predictions to avoid memory issues
                        if len(self.prediction_history) > 50:
                            self.prediction_history = self.prediction_history[-50:]
                        
                        # Send recognized letter to web in real-time
                        if self.full_duplex and self.sio and self.sio.connected:
                            try:
                                emotion_data = {'emotion': self.current_emotion, 'confidence': self.emotion_confidence} if self.current_emotion != "neutral" else None
                                self.sio.emit('sign_recognized', {
                                    'text': maj_label,
                                    'emotion': emotion_data
                                })
                                print(f"[WebSocket] Sent letter: '{maj_label}' with emotion: {emotion_data}")
                            except Exception as e:
                                print(f"[WebSocket] Could not send letter: {e}")
                        
                        if self.per_letter_tts:
                            self._speak(maj_label)
                    
                    # Check for hold duration to add double letters
                    elif self.stable_count > DEBOUNCE_FRAMES and self.letter_added_once and not self.double_letter_added:
                        hold_duration = time.time() - self.letter_hold_start_time
                        # Add double letter after holding for 1 second
                        if hold_duration >= 1.0:
                            self.buffer.append(maj_label)
                            self._last_buffer_activity = time.time()
                            self.double_letter_added = True
                            print(f"[DOUBLE] Added double letter: {maj_label}{maj_label} (hold: {hold_duration:.1f}s)")
                            
                            # Send double letter to web
                            if self.full_duplex and self.sio and self.sio.connected:
                                try:
                                    emotion_data = {'emotion': self.current_emotion, 'confidence': self.emotion_confidence} if self.current_emotion != "neutral" else None
                                    self.sio.emit('sign_recognized', {
                                        'text': maj_label + maj_label,
                                        'emotion': emotion_data
                                    })
                                    print(f"[WebSocket] Sent double letter: '{maj_label}{maj_label}'")
                                except Exception as e:
                                    print(f"[WebSocket] Could not send double letter: {e}")
                    
                    # Update UI with current letter being tracked
                    self.current_letter = maj_label
                else:
                    self.stable_count = 0
            else:
                # Reset tracking when no hand detected
                if self.stable_count > 0:
                    self.letter_hold_start_time = None
                    self.letter_added_once = False
                    self.double_letter_added = False
                self.pred_window.clear()
                self.stable_count = 0
                self.current_letter = None

            # Update resource usage (for metrics tracking, not displayed)
            self.metrics.update_resource_usage()
            
            # End frame timing
            self.metrics.end_frame()

            # Push live metrics to the dashboard
            if self.full_duplex:
                now_t = time.time()
                if (now_t - self._last_metrics_emit_time) >= self._metrics_emit_interval_sec:
                    snapshot = self._build_metrics_snapshot()
                    if self.sio and getattr(self.sio, 'connected', False):
                        try:
                            self.sio.emit('metrics_update', snapshot)
                            self._last_metrics_emit_time = now_t
                            if not self._metrics_emit_ok_logged:
                                print("[WebSocket] ✓ metrics_update emitting to server")
                                self._metrics_emit_ok_logged = True
                        except Exception as e:
                            if not self._metrics_emit_error_logged:
                                print(f"[WebSocket] ✗ metrics_update emit failed: {e}")
                                self._metrics_emit_error_logged = True
                    else:
                        # HTTP fallback to ensure dashboard receives data
                        if requests is not None:
                            try:
                                requests.post('http://localhost:5000/api/metrics_push', json=snapshot, timeout=0.5)
                                self._last_metrics_emit_time = now_t
                                if not self._metrics_emit_ok_logged:
                                    print("[HTTP] ✓ metrics snapshot pushed")
                                    self._metrics_emit_ok_logged = True
                            except Exception as e:
                                if not self._metrics_emit_error_logged:
                                    print(f"[HTTP] ✗ metrics push failed: {e}")
                                    self._metrics_emit_error_logged = True
                        else:
                            if not getattr(self, '_metrics_emit_error_logged', False):
                                print("[HTTP] ✗ requests not installed - cannot push metrics fallback. pip install requests")
                                self._metrics_emit_error_logged = True

                        # Same-process fallback: update websocket_server globals directly (when server is in-thread)
                        try:
                            import websocket_server as _ws
                            _ws.latest_metrics_snapshot = snapshot
                            try:
                                from datetime import datetime as _dt
                                _ws.last_metrics_snapshot_time = _dt.now().isoformat()
                            except Exception:
                                pass
                            if not self._metrics_emit_ok_logged:
                                print("[Direct] ✓ metrics snapshot stored in websocket_server module")
                                self._metrics_emit_ok_logged = True
                        except Exception:
                            # ignore if module not available
                            pass

            # Add word suggestions
            suggestions = self.get_word_suggestions()
            self.draw_suggestion_hud(frame, suggestions)

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32 or key == 13:  # Space or Enter → speak sentence
                text = "".join(self.buffer).strip()
                if text:
                    # Apply grammar correction to the recognized text
                    corrected_text = correct_grammar(text)
                    print(f"[Grammar] Raw: {text}")
                    print(f"[Grammar] Corrected: {corrected_text}")
                    
                    # Store corrected text for display
                    self.corrected_display_text = corrected_text
                    self.display_correction_until = time.time() + 10.0  # Display for 10 seconds
                    
                    # Save to history
                    self.save_to_history(text, corrected_text)
                    
                    # Send full sentence to web via WebSocket (if connected)
                    if self.full_duplex and self.sio and self.sio.connected:
                        try:
                            emotion_data = {'emotion': self.current_emotion, 'confidence': self.emotion_confidence} if self.current_emotion != "neutral" else None
                            self.sio.emit('sign_recognized', {
                                'text': corrected_text,
                                'emotion': emotion_data
                            })
                            print(f"[WebSocket] Sent full sentence: '{corrected_text}' with emotion: {emotion_data}")
                        except Exception as e:
                            print(f"[WebSocket] Could not send full sentence: {e}")
                    self._speak(corrected_text)  # Non-blocking call with corrected text
                self.last_committed = None
            elif key == 8:  # Backspace
                if self.buffer:
                    self.buffer.pop()
                    self._last_buffer_activity = time.time()
            elif key in [ord('1'), ord('2'), ord('3')]:
                idx = int(chr(key)) - 1
                suggestions = self.get_word_suggestions()
                if idx < len(suggestions):
                    target_word = suggestions[idx]
                    
                    # 1. Remove the partial word from the buffer
                    current_text = "".join(self.buffer)
                    last_word_len = len(current_text.split()[-1])
                    for _ in range(last_word_len):
                        if self.buffer:
                            self.buffer.pop()
                            
                    # 2. Add the suggested word + a space
                    for char in target_word:
                        self.buffer.append(char)
                    self.buffer.append(" ")
                    self._last_buffer_activity = time.time()
            elif key == ord("s") or key == ord("S"):
                self.buffer.append(" ")
                self.last_committed = None
                self._last_buffer_activity = time.time()
            elif key == ord("v") or key == ord("V"):
                frame = self._voice_to_sign(frame)
            elif key == ord("h") or key == ord("H"):
                self.show_history()
                self.last_committed = None
            elif key == ord("a") or key == ord("A"):
                self._show_alphabet_signs()
                self.last_committed = None
            elif key == ord("l") or key == ord("L"):
                # Toggle language (L key)
                self._toggle_language()
            elif key == ord("1"):
                # Switch to ASL (1 key)
                self._switch_language("ASL")
            elif key == ord("2"):
                # Switch to ISL (2 key)
                self._switch_language("ISL")

            # Handle voice input button click
            if self.voice_input_requested:
                frame = self._voice_to_sign(frame)
                self.voice_input_requested = False  # Reset flag

            # Auto-send full sentence after a short pause (if full_duplex is enabled) - DISABLED
            # Auto-send functionality disabled to allow unlimited alphabet collection
            # User can manually send with Space/Enter key when ready

        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance metrics report at end of session
        print("\n" + "=" * 70)
        print(self.metrics.get_detailed_report())
        print("=" * 70)


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
        
        # Start WebSocket server in background thread
        def start_websocket_server():
            try:
                from websocket_server import socketio, app
                print("[WebSocket Server] Starting on ws://localhost:5000...")
                socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
            except Exception as e:
                print(f"[WebSocket Server ERROR] {e}")
        
        ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
        ws_thread.start()
        time.sleep(1)  # Give server time to start
        
        RealtimeRecognizer(camera=args.camera, flip=not args.no_flip, per_letter_tts=args.per_letter_tts, 
                  full_duplex=args.full_duplex, auto_send_threshold=args.auto_send_threshold).run()


if __name__ == "__main__":
    main()

