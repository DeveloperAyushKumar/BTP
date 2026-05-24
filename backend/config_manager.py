"""EcoVideo Studio - Backend configuration manager."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DB_PATH = PROJECT_ROOT / "data" / "ecovideo.db"
TRAINED_MODELS_DIR = DATA_DIR / "trained_models"


@dataclass
class PresetConfig:
    """Configuration for a generation preset."""
    name: str
    num_inference_steps: int
    decode_chunk_size: int
    resolution: tuple[int, int]
    attention_slicing: bool = True
    forward_chunking: bool = True
    description: str = ""


# Define the three presets
# SVD native resolution is 1024x576. Must use exact multiples to avoid UNet skip-connection mismatches.
PRESETS: dict[str, PresetConfig] = {
    "eco": PresetConfig(
        name="eco",
        num_inference_steps=8,
        decode_chunk_size=1,
        resolution=(1024, 576),
        attention_slicing=True,
        forward_chunking=True,
        description="Fast & efficient. Minimal VRAM, shortest time.",
    ),
    "balanced": PresetConfig(
        name="balanced",
        num_inference_steps=16,
        decode_chunk_size=1,
        resolution=(1024, 576),
        attention_slicing=True,
        forward_chunking=True,
        description="Good balance of quality and efficiency.",
    ),
    "quality": PresetConfig(
        name="quality",
        num_inference_steps=25,
        decode_chunk_size=1,
        resolution=(1024, 576),
        attention_slicing=True,
        forward_chunking=True,
        description="Best visual quality. More VRAM and time.",
    ),
}


@dataclass
class DeviceConfig:
    """Detected hardware configuration."""
    device: str
    device_name: str
    available_memory_gb: float
    total_memory_gb: float

    @staticmethod
    def detect() -> "DeviceConfig":
        """Auto-detect the best available device."""
        if torch.backends.mps.is_available():
            import psutil
            mem = psutil.virtual_memory()
            config = DeviceConfig(
                device="mps",
                device_name="Apple Silicon (MPS)",
                available_memory_gb=mem.available / (1024**3),
                total_memory_gb=mem.total / (1024**3),
            )
        elif torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            config = DeviceConfig(
                device="cuda",
                device_name=props.name,
                available_memory_gb=torch.cuda.mem_get_info()[0] / (1024**3),
                total_memory_gb=props.total_mem / (1024**3),
            )
        else:
            import psutil
            mem = psutil.virtual_memory()
            config = DeviceConfig(
                device="cpu",
                device_name="CPU",
                available_memory_gb=mem.available / (1024**3),
                total_memory_gb=mem.total / (1024**3),
            )

        logger.info(
            f"Detected device: {config.device_name} | "
            f"Available memory: {config.available_memory_gb:.1f} GB / {config.total_memory_gb:.1f} GB"
        )
        return config


@dataclass
class AppConfig:
    """Application-level configuration."""
    model_id: str = str(Path.home() / ".cache" / "svd-local")
    default_preset: str = "balanced"
    default_seed: int = 42
    num_frames: int = 14
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    output_dir: Path = field(default_factory=lambda: OUTPUTS_DIR)
    carbon_region: str = "global"
    auto_delete_days: int = 30

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "benchmarks").mkdir(parents=True, exist_ok=True)


# Singleton instances
device_config = DeviceConfig.detect()
app_config = AppConfig()

print(f"🌿 EcoVideo Studio | Device: {device_config.device_name} | Memory: {device_config.total_memory_gb:.0f} GB")
