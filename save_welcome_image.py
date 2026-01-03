#!/usr/bin/env python3
"""
Script to save the welcome image from clipboard to welcome_image.png
"""
import sys
from pathlib import Path

try:
    from PIL import ImageGrab
    
    # Grab image from clipboard
    img = ImageGrab.grabclipboard()
    
    if img is None:
        print("[ERROR] No image found in clipboard")
        sys.exit(1)
    
    # Check if it's an image object
    if not hasattr(img, 'save'):
        print("[ERROR] Clipboard content is not an image")
        sys.exit(1)
    
    save_path = Path(__file__).parent / "welcome_image.png"
    img.save(str(save_path))
    print(f"[OK] Image saved to {save_path}")
    
except ImportError:
    print("[ERROR] PIL not installed. Install with: pip install Pillow")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
