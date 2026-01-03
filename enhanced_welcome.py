"""
Enhanced Welcome Image Generator for Sign Language Recognition System
Creates a high-quality welcome image with modern design elements
"""

import cv2
import numpy as np
from pathlib import Path

def create_enhanced_welcome_image(width=800, height=600):
    """Create an enhanced welcome image with gradient background and modern design"""
    
    # Create canvas
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient background (dark blue to lighter blue)
    for y in range(height):
        # Gradient from dark blue (top) to lighter blue (bottom)
        ratio = y / height
        r = int(20 + ratio * 30)  # 20 to 50
        g = int(30 + ratio * 50)  # 30 to 80  
        b = int(60 + ratio * 80)  # 60 to 140
        img[y, :] = [b, g, r]  # OpenCV uses BGR format
    
    # Add decorative circles/elements in background
    for _ in range(15):
        # Random circles with transparency
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        radius = np.random.randint(20, 80)
        color = (
            np.random.randint(100, 200),
            np.random.randint(100, 200), 
            np.random.randint(150, 255)
        )
        cv2.circle(img, (center_x, center_y), radius, color, 2)
    
    # Add main title background panel
    panel_height = 200
    panel_y = height // 2 - panel_height // 2
    panel_margin = 50
    cv2.rectangle(img, 
                  (panel_margin, panel_y), 
                  (width - panel_margin, panel_y + panel_height),
                  (0, 0, 0), -1)
    cv2.rectangle(img, 
                  (panel_margin, panel_y), 
                  (width - panel_margin, panel_y + panel_height),
                  (100, 200, 255), 3)
    
    # Add main title text
    title_text = "Sign ↔ Voice"
    subtitle_text = "Communication System"
    tagline_text = "Breaking Barriers with AI"
    
    # Title
    cv2.putText(img, title_text, 
                (width // 2 - 180, panel_y + 60),
                cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 255), 3, cv2.LINE_AA)
    
    # Subtitle  
    cv2.putText(img, subtitle_text,
                (width // 2 - 200, panel_y + 110),
                cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Tagline
    cv2.putText(img, tagline_text,
                (width // 2 - 150, panel_y + 150),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (100, 255, 100), 2, cv2.LINE_AA)
    
    # Add feature highlights
    features = [
        "• Sign Language Recognition",
        "• Real-time Translation", 
        "• Grammar Correction",
        "• Voice Output"
    ]
    
    start_y = panel_y + panel_height + 40
    for i, feature in enumerate(features):
        y_pos = start_y + (i * 30)
        cv2.putText(img, feature,
                    (panel_margin + 20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Add "Starting..." indicator at bottom
    cv2.putText(img, "Starting...",
                (width // 2 - 80, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Add corner decorations
    corner_size = 50
    corner_color = (0, 255, 255)
    
    # Top-left corner
    cv2.line(img, (0, corner_size), (0, 0), corner_color, 3)
    cv2.line(img, (0, 0), (corner_size, 0), corner_color, 3)
    
    # Top-right corner  
    cv2.line(img, (width - corner_size, 0), (width, 0), corner_color, 3)
    cv2.line(img, (width, 0), (width, corner_size), corner_color, 3)
    
    # Bottom-left corner
    cv2.line(img, (0, height - corner_size), (0, height), corner_color, 3)
    cv2.line(img, (0, height), (corner_size, height), corner_color, 3)
    
    # Bottom-right corner
    cv2.line(img, (width - corner_size, height), (width, height), corner_color, 3)
    cv2.line(img, (width, height), (width, height - corner_size), corner_color, 3)
    
    return img

def save_enhanced_welcome_image():
    """Generate and save the enhanced welcome image"""
    
    # Create high-quality image
    img = create_enhanced_welcome_image(800, 600)
    
    # Save with high quality
    output_path = Path(__file__).parent / "welcome_image.png"
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    print(f"Enhanced welcome image saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    save_enhanced_welcome_image()
