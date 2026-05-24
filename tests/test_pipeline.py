"""Test Pipeline - Verify inference engine basics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image
import numpy as np


def create_test_image(width: int = 512, height: int = 320) -> Image.Image:
    """Create a random test image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestInferenceEngine:
    """Tests for the inference engine (requires model to be loaded)."""

    def test_import(self):
        """Verify inference engine can be imported."""
        from backend.inference_engine import InferenceEngine, GenerationResult
        assert InferenceEngine is not None
        assert GenerationResult is not None

    def test_presets_defined(self):
        """Verify all presets are defined."""
        from backend.config_manager import PRESETS
        assert "eco" in PRESETS
        assert "balanced" in PRESETS
        assert "quality" in PRESETS

    def test_preset_steps(self):
        """Verify preset step counts."""
        from backend.config_manager import PRESETS
        assert PRESETS["eco"].num_inference_steps == 8
        assert PRESETS["balanced"].num_inference_steps == 16
        assert PRESETS["quality"].num_inference_steps == 25

    def test_device_detection(self):
        """Verify device detection works."""
        from backend.config_manager import device_config
        assert device_config.device in ("mps", "cuda", "cpu")
        assert device_config.total_memory_gb > 0


class TestHelpers:
    """Test utility helpers."""

    def test_validate_image_valid(self):
        from utils.helpers import validate_image
        img = create_test_image()
        valid, error = validate_image(img)
        assert valid is True
        assert error is None

    def test_validate_image_too_small(self):
        from utils.helpers import validate_image
        img = Image.new("RGB", (10, 10))
        valid, error = validate_image(img)
        assert valid is False

    def test_format_time(self):
        from utils.helpers import format_time
        assert "s" in format_time(30)
        assert "m" in format_time(90)
