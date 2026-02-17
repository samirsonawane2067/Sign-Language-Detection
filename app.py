"""
Sign Language Recognition - Gradio Web App
Deploy to Hugging Face Spaces for FREE
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import joblib
import tempfile
import os

# Load trained models
try:
    model_asl = joblib.load("models/sign_knn_asl.joblib")
    print("✓ ASL model loaded")
except Exception as e:
    print(f"Warning: Could not load ASL model: {e}")
    model_asl = None

try:
    model_isl = joblib.load("models/sign_knn_isl.joblib")
    print("✓ ISL model loaded")
except Exception as e:
    print(f"Warning: Could not load ISL model: {e}")
    model_isl = None

try:
    model_general = joblib.load("models/sign_knn.joblib")
    print("✓ General model loaded")
except Exception as e:
    print(f"Warning: Could not load general model: {e}")
    model_general = None


def extract_frames(video_path, num_frames=10):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        frame_count += 1
    
    cap.release()
    return frames


def predict_sign(video, language="General"):
    """
    Predict sign language from video input
    
    Args:
        video: Video file path or bytes
        language: "ASL", "ISL", or "General"
    
    Returns:
        Prediction result
    """
    try:
        # Handle different input types
        if isinstance(video, str):
            video_path = video
        elif isinstance(video, tuple):  # Gradio video input
            video_path = video[0] if isinstance(video, tuple) else video
        else:
            return "❌ Invalid video format"
        
        # Select model
        if language == "ASL" and model_asl is not None:
            model = model_asl
        elif language == "ISL" and model_isl is not None:
            model = model_isl
        else:
            model = model_general
        
        if model is None:
            return "❌ Model not loaded. Please check model files."
        
        # Extract frames
        frames = extract_frames(video_path)
        
        if not frames:
            return "❌ Could not extract frames from video"
        
        # Simple prediction (placeholder - adapt to your model)
        predictions = []
        for frame in frames:
            # Resize frame
            frame_resized = cv2.resize(frame, (64, 64))
            frame_flat = frame_resized.flatten().reshape(1, -1)
            
            try:
                pred = model.predict(frame_flat)
                predictions.append(pred[0])
            except Exception as e:
                continue
        
        if not predictions:
            return "❌ Could not make predictions"
        
        # Get most common prediction
        from collections import Counter
        result = Counter(predictions).most_common(1)[0][0]
        
        return f"✓ Recognized Sign: **{result}** (Language: {language})\n\nFrames analyzed: {len(predictions)}"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio Interface
with gr.Blocks(title="Sign Language Recognition") as demo:
    gr.Markdown("""
    # 🤟 Sign Language Recognition
    
    Upload a video of sign language and get the predicted text.
    Supports **ASL** (American Sign Language) and **ISL** (Indian Sign Language).
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Sign Language Video")
            language_choice = gr.Radio(
                ["ASL", "ISL", "General"],
                value="General",
                label="🌍 Select Language"
            )
            submit_btn = gr.Button("🔍 Recognize Sign", variant="primary")
        
        with gr.Column():
            output_text = gr.Markdown(label="Result")
    
    submit_btn.click(
        fn=predict_sign,
        inputs=[video_input, language_choice],
        outputs=output_text
    )
    
    gr.Markdown("""
    ### 📝 How to Use:
    1. Upload a video of sign language
    2. Select the language (ASL, ISL, or General)
    3. Click "Recognize Sign"
    4. Get the prediction result
    
    ### 🎯 Supported Formats:
    - MP4, AVI, MOV, WebM
    - Max duration: 2 minutes
    - Min duration: 2 seconds
    
    ---
    **Built with ❤️ using Sign Language Recognition**
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
