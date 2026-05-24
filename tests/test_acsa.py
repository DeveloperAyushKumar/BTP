"""Test ACSA - Verify classifier and recommender."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from PIL import Image


def create_test_image() -> Image.Image:
    arr = np.random.randint(0, 255, (256, 384, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestACSAClassifier:
    def test_import(self):
        from intelligence.acsa.classifier import ACSAClassifier, PRESET_LABELS
        assert len(PRESET_LABELS) == 3

    def test_classifier_forward(self):
        import torch
        from intelligence.acsa.classifier import ACSAClassifier
        model = ACSAClassifier(input_dim=772)
        x = torch.randn(1, 772)
        out = model(x)
        assert out.shape == (1, 3)

    def test_classifier_predict(self):
        import torch
        from intelligence.acsa.classifier import ACSAClassifier, PRESET_LABELS
        model = ACSAClassifier(input_dim=772)
        x = torch.randn(1, 772)
        preset, confidence = model.predict(x)
        assert preset in PRESET_LABELS
        assert 0.0 <= confidence <= 1.0


class TestHardwareDetector:
    def test_detect(self):
        from intelligence.acsa.hardware_detector import detect_hardware
        hw = detect_hardware()
        assert "gpu_type" in hw
        assert "available_vram_gb" in hw
        assert hw["available_vram_gb"] > 0


class TestRecommenderFallback:
    def test_rule_based(self):
        from intelligence.acsa.recommender import ACSARecommender
        rec = ACSARecommender()
        # Without trained model, should use rule-based
        result = rec._rule_based_recommendation({"available_vram_gb": 10, "compute_score": 0.7, "current_cpu_usage_percent": 20, "total_ram_gb": 18})
        assert result["preset"] in ["eco", "balanced", "quality"]
        assert result["method"] == "rule_based"
