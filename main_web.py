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
HISTORY_FILE = ROOT / "history.json"

MODEL_PATH = MODEL_DIR / "sign_knn.joblib"

WINDOW_TITLE = "Sign Language Recognition"
PRED_SMOOTHING = 7
MIN_CONFIDENCE = 0.6
DEBOUNCE_FRAMES = 8
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
        
        # Grammar correction display
        self.corrected_display_text: Optional[str] = None
        self.display_correction_until: Optional[float] = None
        
        # Emotion detection
        self.emotion_detector = None
        self.current_emotion: str = "neutral"
        self.emotion_confidence: float = 0.0
        try:
            self.emotion_detector = EmotionDetector()
            print("[EMOTION] Emotion detector initialized")
        except Exception as e:
            print(f"[EMOTION] Could not initialize emotion detector: {e}")
        
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
        
        # History storage
        self.history: List[dict] = []
        self.load_history()
        
        # Initialize performance metrics
        self.metrics = PerformanceMetrics(window_size=30)
        
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
                    history_text += f"   Raw: {raw}\n"
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
            img_height = 600
            img_width = 800
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            # Add title
            cv2.putText(img, "HISTORY", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.2, (0, 255, 255), 2, cv2.LINE_AA)
            
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
        
        pred_idx = self.model.predict([feats])[0]
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([feats])[0]
            conf = float(np.max(probs))
        else:
            conf = 1.0
        
        # End processing timer
        self.metrics.end_processing()
        
        # Record prediction for metrics
        label = self.labels[pred_idx]
        self.metrics.record_prediction(label, conf, is_correct=True)
        
        return label, conf

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
        
        # Set window properties for resizing
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        
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
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        extractor = HandFeatureExtractor()

        while True:
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

            # Draw text buffer area below header (adjust for centered frame)
            text_area_y = 120
            # Draw label 'Message:' in green (keep it highlighted)
            label_y = text_area_y + 8
            cv2.putText(frame, 'Message:', (20, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

            # render buffer tokens with colors (letters in white, space as gap) with text wrapping
            x_cursor = 150  # Starting x position
            y_cursor = text_area_y + 8
            line_height = 25  # Height for each line
            max_x = window_width - w - 100  # Maximum x position before video window (with padding)
            
            for token in self.buffer:
                tok = str(token)
                if tok == ' ':
                    # space token - just add spacing, don't draw anything
                    x_cursor += 15
                else:
                    # Check if adding this letter would exceed the video window boundary
                    letter_width = 18  # Approximate width of each letter
                    if x_cursor + letter_width > max_x:
                        # Wrap to next line
                        x_cursor = 150  # Reset to starting x position
                        y_cursor += line_height  # Move to next line
                    
                    # normal letter in white
                    cv2.putText(frame, tok.upper(), (x_cursor, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, (255,255,255), 2, cv2.LINE_AA)
                    x_cursor += letter_width
            
            cv2.putText(frame, f"Pred: {self.current_letter or '-'}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # Draw accuracy below the rendered text buffer
            acc_y = y_cursor + 40
            cv2.putText(frame, acc_text, (20, acc_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
            
            # Display emotion-enhanced text if we have text and emotion
            if self.buffer and self.current_emotion != "neutral":
                emotion_text = "".join(str(t) for t in self.buffer if t != " ")
                enhanced_text = f'"{emotion_text}" ({self.current_emotion.capitalize()})'
                # Position emotion text below accuracy text to prevent overlapping
                emotion_y = acc_y + 40
                cv2.putText(frame, enhanced_text, (20, emotion_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)
            
            # Display corrected text if available and within display time
            if self.corrected_display_text and self.display_correction_until:
                if time.time() < self.display_correction_until:
                    # Calculate corrected text position further below emotion section
                    # Account for emotion text, emotion bar, and more spacing
                    emotion_section_height = 100  # Increased spacing below emotion section
                    corrected_y = emotion_y + emotion_section_height if self.buffer and self.current_emotion != "neutral" else acc_y + emotion_section_height
                    # Draw a background for the corrected text
                    cv2.rectangle(frame, (10, corrected_y - 25), (630, corrected_y + 15), (0, 0, 0), -1)
                    cv2.putText(frame, f"Corrected: {self.corrected_display_text}", (20, corrected_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                else:
                    # Clear the display after timeout
                    self.corrected_display_text = None
                    self.display_correction_until = None

            # Draw controls below the emotion section
            controls_start_y = emotion_y + 180 if self.buffer and self.current_emotion != "neutral" else acc_y + 180
            control_x = 20
            
            # Controls header
            cv2.putText(frame, "CONTROLS:", (control_x, controls_start_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Control items
            controls = [
                ("'S' key", "Add space", (0, 0, 255)),
                ("Space/Enter", "Speak text", (0, 200, 200)),
                ("Backspace", "Delete last", (200, 0, 0)),
                ("'H' key", "Show history", (255, 165, 0)),
                ("'A' key", "Alphabet signs", (255, 255, 255)),
                ("'V' key", "Voice input", (255, 0, 255))
            ]
            for i, (key, action, color) in enumerate(controls, 1):
                y = controls_start_y + (i * 25)
                cv2.putText(frame, f"{key}:", (control_x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{action}", (control_x + 120, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Process hand landmarks with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                res = hands.process(rgb)
            except Exception as e:
                print(f"[MediaPipe] Error processing frame: {e}")
                res = None  # Continue with None result

            # Process emotion detection if available
            if self.emotion_detector:
                try:
                    self.current_emotion, self.emotion_confidence = self.emotion_detector.detect_emotion(frame)
                    # Position emotion info below all text elements
                    emotion_start_y = emotion_y + 40 if self.buffer and self.current_emotion != "neutral" else acc_y + 40
                    frame = self.emotion_detector.draw_emotion_info(frame, self.current_emotion, self.emotion_confidence, emotion_start_y)
                except Exception as e:
                    print(f"[EMOTION] Error detecting emotion: {e}")

            # Process hand landmarks if any
            if res.multi_hand_landmarks:
                # Draw hand landmarks on frame (adjust for centered frame)
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
                    
                    # Check for hold duration to add extra letters
                    elif self.letter_added_once and self.letter_hold_start_time:
                        hold_time = time.time() - self.letter_hold_start_time
                        
                        # Add letters every 0.5 seconds up to 10 total letters
                        max_extra_letters = 9  # 1 initial + 9 extra = 10 total
                        interval = 0.5  # Add letter every 0.5 seconds
                        
                        for i in range(max_extra_letters):
                            expected_time = (i + 1) * interval
                            if hold_time > expected_time and self.extra_letters_added == i:
                                self.buffer.append(maj_label)
                                self.extra_letters_added = i + 1
                                self._last_buffer_activity = time.time()
                                
                                if self.full_duplex and self.sio and self.sio.connected:
                                    try:
                                        emotion_data = {'emotion': self.current_emotion, 'confidence': self.emotion_confidence} if self.current_emotion != "neutral" else None
                                        self.sio.emit('sign_recognized', {
                                            'text': maj_label,
                                            'emotion': emotion_data
                                        })
                                        print(f"[WebSocket] Sent extra letter ({expected_time}s): '{maj_label}' with emotion: {emotion_data}")
                                    except Exception as e:
                                        pass
                                break  # Only add one letter per frame
            else:
                # Hand gesture ended, reset tracking
                self.pred_window.clear()
                self.stable_count = 0
                self.letter_hold_start_time = None
                self.letter_added_once = False
                self.extra_letters_added = 0

            # Update resource usage (for metrics tracking, not displayed)
            self.metrics.update_resource_usage()
            
            # End frame timing
            self.metrics.end_frame()

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

