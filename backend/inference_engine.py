"""Inference Engine - Orchestrates video generation."""

from __future__ import annotations

import gc
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from backend.config_manager import PRESETS, AppConfig, PresetConfig, app_config, device_config
from ml_components.svd_loader import SVDLoader

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a video generation."""
    generation_id: str
    frames: list[Image.Image]
    generation_time_seconds: float
    peak_vram_gb: float
    preset_name: str
    resolution: tuple[int, int]
    num_steps: int
    seed: int
    output_path: Optional[Path] = None


class InferenceEngine:
    """Main inference engine for video generation."""

    def __init__(self, loader: Optional[SVDLoader] = None):
        self.loader = loader or SVDLoader()
        self._pipeline = None

    def ensure_model_loaded(self):
        """Ensure the SVD pipeline is loaded."""
        if not self.loader.is_loaded:
            self._pipeline = self.loader.load()
        else:
            self._pipeline = self.loader.pipeline

    def generate(
        self,
        image: Image.Image,
        preset_name: str = "balanced",
        seed: int = 42,
        num_frames: int = 14,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> GenerationResult:
        """
        Generate a video from an input image.

        Args:
            image: Input PIL Image
            preset_name: One of 'eco', 'balanced', 'quality'
            seed: Random seed for reproducibility
            num_frames: Number of output frames
            motion_bucket_id: Motion intensity (1-255)
            noise_aug_strength: Noise augmentation strength
            progress_callback: Callable(step, total_steps, elapsed_time)

        Returns:
            GenerationResult with frames and metadata
        """
        self.ensure_model_loaded()
        preset = PRESETS[preset_name]
        self.loader.apply_preset(preset_name)

        # Let the pipeline handle resizing internally — don't override height/width
        # Just ensure image is RGB
        image_resized = image.convert("RGB")

        # Setup generator for reproducibility
        generator = torch.Generator(device=device_config.device).manual_seed(seed)

        # Track metrics
        start_time = time.time()

        # Step callback for progress
        step_count = [0]

        def step_callback_fn(pipe, step_index, timestep, callback_kwargs):
            step_count[0] = step_index + 1
            elapsed = time.time() - start_time
            if progress_callback:
                progress_callback(step_index + 1, preset.num_inference_steps, elapsed)
            return callback_kwargs

        # Run inference
        logger.info(f"Generating video: preset={preset_name}, steps={preset.num_inference_steps}, seed={seed}, frames={num_frames}")

        # Limit frame count per preset to fit in memory
        max_frames = {"eco": 4, "balanced": 6, "quality": 14}
        num_frames = min(num_frames, max_frames.get(preset_name, 6))

        with torch.no_grad():
            output = self._pipeline(
                image=image_resized,
                decode_chunk_size=preset.decode_chunk_size,
                generator=generator,
                num_frames=num_frames,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                num_inference_steps=preset.num_inference_steps,
                callback_on_step_end=step_callback_fn,
            )

        generation_time = time.time() - start_time

        # Get peak memory
        peak_vram_gb = 0.0
        try:
            import psutil
            peak_vram_gb = psutil.virtual_memory().used / (1024**3)
        except Exception:
            pass

        # Extract frames
        frames = output.frames[0]  # List of PIL Images

        # Cleanup
        gc.collect()
        if device_config.device == "mps":
            torch.mps.empty_cache()
        elif device_config.device == "cuda":
            torch.cuda.empty_cache()

        generation_id = str(uuid.uuid4())[:8]

        result = GenerationResult(
            generation_id=generation_id,
            frames=frames,
            generation_time_seconds=generation_time,
            peak_vram_gb=peak_vram_gb,
            preset_name=preset_name,
            resolution=preset.resolution,
            num_steps=preset.num_inference_steps,
            seed=seed,
        )

        logger.info(
            f"Generation complete: {generation_time:.1f}s, "
            f"peak VRAM: {peak_vram_gb:.2f} GB, preset: {preset_name}"
        )
        return result

    def save_video(self, result: GenerationResult, output_dir: Optional[Path] = None) -> Path:
        """Save generated frames as an MP4 video."""
        from diffusers.utils import export_to_video

        output_dir = output_dir or app_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"ecovideo_{result.generation_id}_{result.preset_name}.mp4"
        output_path = output_dir / filename

        export_to_video(result.frames, str(output_path), fps=7)

        result.output_path = output_path
        logger.info(f"Video saved: {output_path}")
        return output_path
