"""
Sign Language Recognition - Streamlit Web App
Deploy to Streamlit Cloud for FREE

Run locally: streamlit run app_streamlit.py
Deploy: Push to GitHub → Connect to Streamlit Cloud
"""

import streamlit as st
import cv2
import numpy as np
import joblib
from pathlib import Path
from collections import Counter
import tempfile

# Page config
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🤟 Sign Language Recognition")
st.write("Upload a video to recognize sign language in real-time")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    language = st.radio(
        "Select Language",
        ["General", "ASL", "ISL"],
        help="ASL: American Sign Language | ISL: Indian Sign Language"
    )
    num_frames = st.slider("Frames to Analyze", 5, 20, 10)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    model_paths = {
        "ASL": "models/sign_knn_asl.joblib",
        "ISL": "models/sign_knn_isl.joblib",
        "General": "models/sign_knn.joblib"
    }
    
    for name, path in model_paths.items():
        try:
            if Path(path).exists():
                models[name] = joblib.load(path)
                st.sidebar.success(f"✓ {name} model loaded")
            else:
                st.sidebar.warning(f"⚠ {name} model not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {name}: {e}")
    
    return models

models = load_models()

if not models:
    st.error("❌ No models loaded! Please check model files.")
    st.stop()

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("📹 Upload Video")
    video_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "webm"],
        help="Upload a video of sign language"
    )

with col2:
    st.subheader("🔍 Results")
    result_placeholder = st.empty()

# Process video
if video_file is not None:
    # Display video
    st.video(video_file)
    
    # Process button
    if st.button("🎯 Recognize Sign Language", key="process_btn", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(video_file.read())
                video_path = tmp_file.name
            
            # Extract frames
            status_text.info("📊 Extracting frames...")
            progress_bar.progress(20)
            
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
            
            if not frames:
                st.error("❌ Could not extract frames from video")
            else:
                progress_bar.progress(40)
                status_text.info("🔍 Analyzing frames...")
                
                # Get model
                model = models.get(language)
                if model is None:
                    st.error(f"❌ {language} model not available")
                else:
                    # Make predictions
                    predictions = []
                    for i, frame in enumerate(frames):
                        try:
                            # Preprocess frame
                            frame_resized = cv2.resize(frame, (64, 64))
                            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                            frame_flat = frame_gray.flatten().reshape(1, -1)
                            
                            # Predict
                            pred = model.predict(frame_flat)
                            predictions.append(pred[0])
                            
                            progress = 40 + (i / len(frames)) * 50
                            progress_bar.progress(int(progress))
                        except Exception as e:
                            continue
                    
                    progress_bar.progress(100)
                    status_text.success("✓ Analysis complete!")
                    
                    if predictions:
                        # Get most common prediction
                        result = Counter(predictions).most_common(1)[0][0]
                        confidence = (Counter(predictions)[result] / len(predictions)) * 100
                        
                        # Display result
                        with result_placeholder.container():
                            st.markdown(f"## ✓ **{result}**")
                            st.metric("Language", language)
                            st.metric("Confidence", f"{confidence:.1f}%")
                            st.metric("Frames Analyzed", len(predictions))
                            
                            # Show detailed stats
                            with st.expander("📊 Detailed Statistics"):
                                st.write("**Prediction Distribution:**")
                                pred_counts = Counter(predictions)
                                st.bar_chart(pred_counts)
                    else:
                        st.error("❌ Could not make predictions from frames")
        
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
        
        finally:
            # Cleanup
            try:
                Path(video_path).unlink()
            except:
                pass

# Information section
with st.expander("ℹ️ About This App"):
    st.markdown("""
    ### 🤟 Sign Language Recognition System
    
    This app uses machine learning to recognize sign language from videos.
    
    **Supported Languages:**
    - 🇺🇸 **ASL** - American Sign Language
    - 🇮🇳 **ISL** - Indian Sign Language
    - 🌍 **General** - Multi-language recognition
    
    **How it Works:**
    1. Upload a sign language video
    2. The system extracts key frames
    3. Machine learning model analyzes each frame
    4. Returns the recognized sign and confidence score
    
    **Tips for Best Results:**
    - 📹 Clear lighting and good video quality
    - ✋ Hand should be clearly visible
    - 🎯 Focus on one sign per video
    - ⏱️ Video duration: 2-10 seconds
    
    ---
    **Built with Streamlit & Scikit-learn**
    """)

# Footer
st.markdown("""
    ---
    <div style='text-align: center'>
    <p>🤟 Sign Language Recognition v1.0</p>
    <p>Made with ❤️ for accessibility</p>
    </div>
""", unsafe_allow_html=True)
