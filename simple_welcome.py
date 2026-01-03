"""
Simple Original Welcome Image Generator for Sign Language Recognition System
Creates a simple, clean welcome image similar to the original style
"""

import cv2
import numpy as np
from pathlib import Path

def create_simple_welcome_image(width=640, height=480):
    """Create a simple welcome image with basic design"""
    
    # Create dark background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # Dark gray background
    
    # Add simple title
    title_text = "Welcome to Sign Voice"
    subtitle_text = "Communication System"
    starting_text = "Starting..."
    
    # Title
    cv2.putText(img, title_text, 
                (int(width*0.1), int(height*0.35)),
                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    
    # Subtitle  
    cv2.putText(img, subtitle_text,
                (int(width*0.1), int(height*0.45)),
                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    
    # Starting text
    cv2.putText(img, starting_text,
                (int(width*0.1), int(height*0.60)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 1, cv2.LINE_AA)
    
    return img

def save_simple_welcome_image():
    """Generate and save the simple welcome image"""
    
    # Create simple image
    img = create_simple_welcome_image(640, 480)
    
    # Save with high quality
    output_path = Path(__file__).parent / "welcome_image.png"
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    print(f"Simple welcome image saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    save_simple_welcome_image()
