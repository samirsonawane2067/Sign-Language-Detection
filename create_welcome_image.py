#!/usr/bin/env python3
"""
Create a welcome image for the Sign Language Recognition app
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np

# Create image with gradient background (blue to cyan)
width, height = 1024, 576
img = Image.new('RGB', (width, height), color=(100, 150, 200))

# Create a gradient effect
pixels = img.load()
for y in range(height):
    # Gradient from blue-purple to cyan
    r = int(100 + (150 - 100) * (y / height))
    g = int(150 + (200 - 150) * (y / height))
    b = int(200 + (255 - 200) * (y / height))
    for x in range(width):
        pixels[x, y] = (r, g, b)

draw = ImageDraw.Draw(img, 'RGBA')

# Try to use a nice font, fall back to default
try:
    title_font = ImageFont.truetype("arial.ttf", 80)
    subtitle_font = ImageFont.truetype("arial.ttf", 60)
    text_font = ImageFont.truetype("arial.ttf", 32)
except:
    title_font = ImageFont.load_default()
    subtitle_font = ImageFont.load_default()
    text_font = ImageFont.load_default()

# Draw title
title = "Welcome to"
draw.text((width//2 - 200, height//2 - 150), title, fill=(255, 255, 255), font=title_font)

# Draw subtitle
subtitle = "Sign Language Recognition"
draw.text((width//2 - 350, height//2 - 50), subtitle, fill=(100, 200, 255), font=subtitle_font)

# Draw instruction text
instruction = "Get ready to sign! The camera will open"
draw.text((width//2 - 250, height//2 + 100), instruction, fill=(50, 50, 50), font=text_font)

instruction2 = "shortly to interpret your gestures."
draw.text((width//2 - 200, height//2 + 150), instruction2, fill=(50, 50, 50), font=text_font)

# Draw camera icon placeholder
camera_x, camera_y = width//2, height//2 + 250
draw.ellipse([camera_x - 30, camera_y - 30, camera_x + 30, camera_y + 30], outline=(200, 200, 200), width=3)
draw.ellipse([camera_x - 15, camera_y - 15, camera_x + 15, camera_y + 15], outline=(200, 200, 200), width=2)

# Save image
save_path = Path(__file__).parent / "welcome_image.png"
img.save(str(save_path))
print(f"[OK] Welcome image created at {save_path}")
