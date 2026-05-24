"""ACSA Recommender - Main entry point for preset recommendation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from backend.config_manager import TRAINED_MODELS_DIR
from intelligence.acsa.classifier import ACSAClassifier
from intelligence.acsa.feature_extractor import FeatureExtractor
from intelligence.acsa.hardware_detector import detect_hardware

logger = logging.getLogger(__name__)

MODEL_PATH = TRAINED_MODELS_DIR / "acsa_classifier.pth"


class ACSARecommender:
    """Recommends optimal preset using CLIP features + hardware profile."""

    def __init__(self):
        self._classifier = ACSAClassifier()
        self._extractor = FeatureExtractor()
        self._model_loaded = self._classifier.load(MODEL_PATH)

    def recommend_preset(self, image: Image.Image) -> dict:
        """
        Recommend the optimal preset for an image.
        
        Returns:
            Dict with 'preset', 'confidence', 'method' keys.
        """
        hw = detect_hardware()

        if self._model_loaded:
            return self._ml_recommendation(image, hw)
        else:
            return self._rule_based_recommendation(hw)

    def _ml_recommendation(self, image: Image.Image, hw: dict) -> dict:
        """Use trained classifier for recommendation."""
        # Extract CLIP features
        clip_features = self._extractor.extract(image)  # (1, 768)

        # Concatenate hardware features
        hw_tensor = torch.tensor([[
            hw["available_vram_gb"],
            hw["compute_score"],
            hw["current_cpu_usage_percent"] / 100.0,
            hw["total_ram_gb"] / 64.0,  # Normalize
        ]], dtype=torch.float32)

        features = torch.cat([clip_features, hw_tensor], dim=1)  # (1, 772)

        preset, confidence = self._classifier.predict(features)

        return {
            "preset": preset,
            "confidence": round(confidence, 3),
            "method": "ml_classifier",
            "hardware": hw,
        }

    def _rule_based_recommendation(self, hw: dict) -> dict:
        """Fallback rule-based recommendation."""
        vram = hw["available_vram_gb"]

        if vram < 8:
            preset = "eco"
        elif vram < 16:
            preset = "balanced"
        else:
            preset = "quality"

        return {
            "preset": preset,
            "confidence": 0.7,
            "method": "rule_based",
            "hardware": hw,
        }
