"""Quality Predictor - Pre-generation quality estimates."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class QualityPredictor:
    """Predicts expected quality before generation based on image properties."""

    @staticmethod
    def analyze_image(image: Image.Image) -> dict:
        """Analyze input image properties that affect generation quality."""
        arr = np.array(image.convert("RGB"))

        # Image complexity (edge density)
        gray = np.mean(arr, axis=2)
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        edge_density = (np.mean(dx) + np.mean(dy)) / 2.0

        # Color variance
        color_std = np.std(arr, axis=(0, 1)).mean()

        # Brightness
        brightness = np.mean(gray) / 255.0

        # Contrast
        contrast = np.std(gray) / 255.0

        return {
            "edge_density": float(edge_density),
            "color_variance": float(color_std),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "resolution": image.size,
        }

    @staticmethod
    def predict_quality(image: Image.Image, preset_name: str) -> dict:
        """Predict expected quality for a given preset."""
        props = QualityPredictor.analyze_image(image)

        # Heuristic prediction based on image properties and preset
        base_scores = {"eco": 0.82, "balanced": 0.88, "quality": 0.93}
        base = base_scores.get(preset_name, 0.88)

        # High complexity images tend to have lower quality with fewer steps
        complexity_penalty = max(0, (props["edge_density"] - 20) * 0.002)
        predicted_ssim = base - complexity_penalty

        return {
            "predicted_ssim": round(predicted_ssim, 3),
            "image_complexity": "high" if props["edge_density"] > 25 else "medium" if props["edge_density"] > 15 else "low",
            "recommendation": "Use Quality preset" if props["edge_density"] > 25 else "Balanced preset is suitable",
        }
