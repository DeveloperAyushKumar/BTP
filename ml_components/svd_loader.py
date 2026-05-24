"""SVD Model Loader with MPS optimizations."""

from __future__ import annotations

import gc
import logging
import time
from typing import Optional

import torch
from diffusers import StableVideoDiffusionPipeline

from backend.config_manager import (
    PRESETS,
    AppConfig,
    DeviceConfig,
    PresetConfig,
    app_config,
    device_config,
)

logger = logging.getLogger(__name__)


class SVDLoader:
    """Loads and configures the Stable Video Diffusion pipeline for MPS."""

    def __init__(self, config: AppConfig = app_config, device: DeviceConfig = device_config):
        self.config = config
        self.device = device
        self.pipeline: Optional[StableVideoDiffusionPipeline] = None
        self._loaded = False

    def load(self) -> StableVideoDiffusionPipeline:
        """Load the SVD model and move to device (MPS/CUDA/CPU)."""
        if self._loaded and self.pipeline is not None:
            return self.pipeline

        logger.info(f"Loading SVD model: {self.config.model_id}")
        start = time.time()

        # Load from local directory — files are already downloaded
        # Use float16 to save memory (the safetensors files are fp16 weights)
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        # Move to device
        self.pipeline = self.pipeline.to(self.device.device)

        # Apply M3 Pro optimizations
        self._apply_optimizations()

        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s on {self.device.device_name}")
        self._loaded = True
        return self.pipeline

    def _apply_optimizations(self):
        """Apply MPS-compatible memory optimizations."""
        if self.pipeline is None:
            return

        # NOTE: attention_slicing and forward_chunking are NOT compatible 
        # with SVD's spatio-temporal UNet on MPS - they corrupt tensor dimensions
        # Memory is managed via decode_chunk_size and PYTORCH_MPS_HIGH_WATERMARK_RATIO instead

        logger.info("Model loaded with default settings (no slicing/chunking for MPS compatibility)")

    def apply_preset(self, preset_name: str):
        """Apply a preset's optimization settings to the pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        preset = PRESETS[preset_name]

        if preset.attention_slicing:
            self.pipeline.enable_attention_slicing()
        else:
            self.pipeline.disable_attention_slicing()

    def unload(self):
        """Unload the model to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._loaded = False
            gc.collect()
            if self.device.device == "mps":
                torch.mps.empty_cache()
            elif self.device.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded, memory freed.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded
