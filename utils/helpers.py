"""Helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image


def validate_image(image: Image.Image, max_size: int = 4096) -> tuple[bool, Optional[str]]:
    """Validate an input image for generation."""
    if image is None:
        return False, "No image provided"
    
    w, h = image.size
    if w < 64 or h < 64:
        return False, f"Image too small: {w}x{h}. Minimum 64x64."
    if w > max_size or h > max_size:
        return False, f"Image too large: {w}x{h}. Maximum {max_size}x{max_size}."
    
    if image.mode not in ("RGB", "RGBA", "L"):
        return False, f"Unsupported image mode: {image.mode}"
    
    # Convert to RGB if needed
    return True, None


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB mode."""
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode != "RGB":
        return image.convert("RGB")
    return image


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def format_bytes(bytes_val: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"
