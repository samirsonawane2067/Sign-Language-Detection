"""
Emotion Detection Module using MediaPipe FaceMesh
Detects basic emotions: happy, sad, angry, neutral
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import time

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Install with: pip install mediapipe")

class EmotionDetector:
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is required for emotion detection")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Emotion state tracking
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        self.emotion_history = []
        self.last_emotion_time = 0
        self.emotion_stable_frames = 0
        self.EMOTION_STABILITY_THRESHOLD = 8  # Increased from 5 for more stability
        self.EMOTION_CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence to accept emotion
        
        # Face mesh landmark indices for key facial features
        # Mouth points for smile detection
        self.mouth_outer = [61, 84, 17, 314, 405, 291, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        self.mouth_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        
        # Eye points for squinting/anger detection
        self.left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        
        # Eyebrow points for anger/sadness detection
        self.left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        self.right_eyebrow = [296, 334, 293, 300, 276, 283, 282, 295, 285, 336]

    def calculate_mouth_curve(self, landmarks) -> float:
        """Fixed mouth curvature calculation - positive = smile, negative = frown"""
        try:
            # Get key mouth points
            mouth_left = landmarks[61]  # Left corner
            mouth_right = landmarks[291]  # Right corner
            mouth_top = landmarks[13]  # Top center (upper lip)
            mouth_bottom = landmarks[14]  # Bottom center (lower lip)
            mouth_left_upper = landmarks[78]  # Left upper lip
            mouth_right_upper = landmarks[308]  # Right upper lip
            
            # Calculate mouth width for normalization
            mouth_width = abs(mouth_right.x - mouth_left.x)
            
            # FIXED: Calculate smile/frown curve correctly
            # For a smile: corners should be HIGHER (smaller y) than the lip center
            # For a frown: corners should be LOWER (larger y) than the lip center
            
            corner_avg_y = (mouth_left.y + mouth_right.y) / 2
            lip_center_y = (mouth_top.y + mouth_bottom.y) / 2
            
            # FIXED: Positive = smile (corners above center), Negative = frown (corners below center)
            raw_curve = lip_center_y - corner_avg_y  # Reversed the calculation
            
            # Normalize by mouth width to account for face size/distance
            normalized_curve = raw_curve / (mouth_width + 1e-6)
            
            # Debug output (can be commented out in production)
            # print(f"Debug - Corner Y: {corner_avg_y:.4f}, Center Y: {lip_center_y:.4f}, Raw: {raw_curve:.4f}, Normalized: {normalized_curve:.4f}")
            
            return normalized_curve
        except Exception as e:
            print(f"Error in mouth curve calculation: {e}")
            return 0.0

    def calculate_eye_aspect_ratio(self, landmarks, eye_points) -> float:
        """Calculate eye aspect ratio for squinting detection"""
        try:
            eye_points = [landmarks[i] for i in eye_points]
            
            # Calculate eye height (vertical distance)
            eye_height1 = np.linalg.norm([eye_points[1].x - eye_points[5].x, eye_points[1].y - eye_points[5].y])
            eye_height2 = np.linalg.norm([eye_points[2].x - eye_points[4].x, eye_points[2].y - eye_points[4].y])
            eye_height = (eye_height1 + eye_height2) / 2.0
            
            # Calculate eye width (horizontal distance)
            eye_width = np.linalg.norm([eye_points[0].x - eye_points[3].x, eye_points[0].y - eye_points[3].y])
            
            # Eye aspect ratio
            ear = eye_height / (eye_width + 1e-6)
            return ear
        except:
            return 0.3  # Normal eye aspect ratio

    def calculate_eyebrow_position(self, landmarks, eyebrow_points) -> float:
        """FIXED eyebrow position calculation"""
        try:
            eyebrow_points = [landmarks[i] for i in eyebrow_points]
            eyebrow_avg_y = np.mean([p.y for p in eyebrow_points])
            
            # FIXED: Use simple eye center reference
            if eyebrow_points == self.left_eyebrow:
                eye_center = landmarks[33].y  # Left eye center
            else:  # right_eyebrow
                eye_center = landmarks[362].y  # Right eye center
            
            # FIXED: Simple calculation - positive = raised, negative = furrowed
            eyebrow_height = eye_center - eyebrow_avg_y  # Reversed calculation
            
            return eyebrow_height
        except Exception as e:
            print(f"Error in eyebrow calculation: {e}")
            return 0.0

    def classify_emotion(self, mouth_curve: float, left_ear: float, right_ear: float, 
                        left_eyebrow: float, right_eyebrow: float) -> Tuple[str, float]:
        """Final emotion classification with corrected calculations"""
        
        # Average values
        avg_ear = (left_ear + right_ear) / 2
        avg_eyebrow = (left_eyebrow + right_eyebrow) / 2
        
        # Initialize emotion scores
        emotions = {}
        
        # HAPPY - Clear smile
        if mouth_curve > 0.015:  # Clear smile
            emotions['happy'] = 0.8
        
        # SAD - Clear frown
        elif mouth_curve < -0.012:  # Clear frown
            emotions['sad'] = 0.8
        
        # ANGRY - Furrowed eyebrows (only if clearly furrowed)
        elif avg_eyebrow < -0.015:  # Clearly furrowed eyebrows
            emotions['angry'] = 0.7
        
        # NEUTRAL - Everything else
        else:
            emotions['neutral'] = 0.6
        
        # Return emotion with highest confidence
        if emotions:
            emotion = max(emotions, key=emotions.get)
            confidence = emotions[emotion]
            return emotion, confidence
        
        return "neutral", 0.5

    def detect_emotion(self, frame: np.ndarray) -> Tuple[str, float]:
        """Main emotion detection function"""
        if not MEDIAPIPE_AVAILABLE:
            return "neutral", 0.0
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return self.current_emotion, 0.0
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract facial features
        mouth_curve = self.calculate_mouth_curve(face_landmarks.landmark)
        left_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, self.left_eye)
        right_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, self.right_eye)
        left_eyebrow = self.calculate_eyebrow_position(face_landmarks.landmark, self.left_eyebrow)
        right_eyebrow = self.calculate_eyebrow_position(face_landmarks.landmark, self.right_eyebrow)
        
        # Classify emotion
        emotion, confidence = self.classify_emotion(mouth_curve, left_ear, right_ear, left_eyebrow, right_eyebrow)
        
        # Apply temporal smoothing to reduce flickering and improve accuracy
        self.emotion_history.append((emotion, confidence))
        if len(self.emotion_history) > 15:  # Increased history size for better smoothing
            self.emotion_history.pop(0)
        
        # Count emotion occurrences in recent history with confidence weighting
        emotion_scores = {}
        for hist_emotion, hist_conf in self.emotion_history:
            if hist_emotion not in emotion_scores:
                emotion_scores[hist_emotion] = []
            # Only consider emotions that meet minimum confidence threshold
            if hist_conf >= self.EMOTION_CONFIDENCE_THRESHOLD:
                emotion_scores[hist_emotion].append(hist_conf)
        
        # Find most stable emotion with sufficient frames and confidence
        if emotion_scores:
            # Calculate weighted scores for each emotion
            best_emotion = "neutral"
            best_score = 0
            
            for emotion, confidences in emotion_scores.items():
                if len(confidences) >= self.EMOTION_STABILITY_THRESHOLD:
                    avg_confidence = np.mean(confidences)
                    stability_bonus = len(confidences) / self.EMOTION_STABILITY_THRESHOLD
                    final_score = avg_confidence * stability_bonus
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_emotion = emotion
            
            # Only update if we have a stable emotion with good confidence
            if best_emotion != "neutral" and best_score > 0.5:
                self.current_emotion = best_emotion
                self.emotion_confidence = min(best_score, 1.0)
            elif best_emotion == "neutral" and len(emotion_scores.get("neutral", [])) >= self.EMOTION_STABILITY_THRESHOLD:
                self.current_emotion = "neutral"
                self.emotion_confidence = np.mean(emotion_scores["neutral"])
        
        return self.current_emotion, self.emotion_confidence

    def draw_emotion_info(self, frame: np.ndarray, emotion: str, confidence: float, start_y: int = 120) -> np.ndarray:
        """Draw emotion information on frame at specified position"""
        # Emotion color coding
        emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'neutral': (128, 128, 128) # Gray
        }
        
        color = emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw emotion text at specified position
        emotion_text = f"Emotion: {emotion.capitalize()} ({confidence:.2f})"
        cv2.putText(frame, emotion_text, (20, start_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color, 2, cv2.LINE_AA)
        
        # Draw emotion indicator bar below text
        bar_y = start_y + 20
        cv2.rectangle(frame, (20, bar_y), (20 + int(confidence * 200), bar_y + 20), color, -1)
        cv2.rectangle(frame, (20, bar_y), (220, bar_y + 20), (255, 255, 255), 2)
        
        return frame

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
