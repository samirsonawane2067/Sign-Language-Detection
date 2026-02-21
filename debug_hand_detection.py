#!/usr/bin/env python3
"""
Debug script to diagnose hand detection and prediction issues.
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path

def debug_hand_detection():
    """Debug hand detection and model prediction."""
    
    print("🔍 Debugging Hand Detection and Prediction")
    print("=" * 50)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Load models
    models = {}
    model_files = {
        'ASL': 'models/sign_knn_asl.joblib',
        'ISL': 'models/sign_knn_isl.joblib',
        'ISL_Dual': 'models/sign_knn_isl_dual_hands.joblib',
        'General': 'models/sign_knn.joblib'
    }
    
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
            print(f"✅ Loaded {name} model from {path}")
            if hasattr(models[name], 'classes_'):
                print(f"   Classes: {list(models[name].classes_)}")
        except Exception as e:
            print(f"❌ Failed to load {name} model: {e}")
    
    # Feature extractor
    class HandFeatureExtractor:
        def from_mediapipe(self, hand_landmarks):
            """Extract features from MediaPipe hand landmarks."""
            if not hand_landmarks:
                return None
            
            features = []
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(features)
    
    extractor = HandFeatureExtractor()
    
    print("\n🎯 Starting camera debug...")
    print("Show your hand to the camera and press 'q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)  # Mirror image
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Extract features and predict
                try:
                    features = extractor.from_mediapipe(hand_landmarks)
                    if features is not None:
                        print(f"\n📊 Frame {frame_count}: Hand detected!")
                        print(f"   Feature vector length: {len(features)}")
                        print(f"   Feature sample: {features[:5]}")
                        
                        # Test prediction with each model
                        for name, model in models.items():
                            try:
                                if hasattr(model, 'predict'):
                                    pred = model.predict([features])[0]
                                    if hasattr(model, 'predict_proba'):
                                        conf = model.predict_proba([features]).max()
                                        print(f"   {name}: '{pred}' (confidence: {conf:.3f})")
                                    else:
                                        print(f"   {name}: '{pred}' (no confidence)")
                            except Exception as e:
                                print(f"   {name}: Error - {e}")
                        
                except Exception as e:
                    print(f"❌ Feature extraction error: {e}")
        else:
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"📷 Frame {frame_count}: No hands detected")
        
        # Display info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if results.multi_hand_landmarks:
            cv2.putText(frame, "HAND DETECTED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO HANDS", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Hand Detection Debug', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Debug session completed")

if __name__ == "__main__":
    debug_hand_detection()
