#!/usr/bin/env python3
"""
Test script for emotion detection functionality
"""

import cv2
import time
from emotion_detector import EmotionDetector

def test_emotion_detection():
    """Test emotion detection with webcam"""
    print("Starting emotion detection test...")
    print("Make different facial expressions:")
    print("- Smile for 'happy'")
    print("- Frown for 'sad'")
    print("- Furrow brows for 'angry'")
    print("- Neutral face for 'neutral'")
    print("\nPress 'q' to quit")
    
    # Initialize emotion detector
    try:
        detector = EmotionDetector()
        print("✓ Emotion detector initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize emotion detector: {e}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot access webcam")
        return
    
    print("✓ Webcam initialized")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect emotion
        emotion, confidence = detector.detect_emotion(frame)
        
        # Draw emotion info
        frame = detector.draw_emotion_info(frame, emotion, confidence)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display frame
        cv2.imshow("Emotion Detection Test", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
    print("✓ Test completed")

if __name__ == "__main__":
    test_emotion_detection()
