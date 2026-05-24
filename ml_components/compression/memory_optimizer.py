"""Compression - Memory optimizer for MPS."""

from __future__ import annotations

import gc
import logging

import torch

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Applies memory optimization techniques for Apple Silicon."""

    @staticmethod
    def apply_attention_slicing(pipeline, slice_size: str = "auto"):
        """Enable attention slicing to reduce peak VRAM."""
        pipeline.enable_attention_slicing(slice_size)
        logger.info(f"Attention slicing enabled: {slice_size}")

    @staticmethod
    def apply_forward_chunking(pipeline, chunk_size: int = 1, dim: int = 1):
        """Enable forward chunking on UNet."""
        pipeline.unet.enable_forward_chunking(chunk_size=chunk_size, dim=dim)
        logger.info(f"Forward chunking enabled: chunk_size={chunk_size}")

    @staticmethod
    def clear_memory(device: str = "mps"):
        """Aggressively free memory."""
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        logger.debug("Memory cleared")

    @staticmethod
    def get_memory_usage(device: str = "mps") -> dict:
        """Get current memory usage stats."""
        if device == "mps":
            try:
                allocated = torch.mps.driver_allocated_memory() / (1024**3)
                return {"allocated_gb": allocated}
            except Exception:
                return {"allocated_gb": 0.0}
        elif device == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            }
        return {"allocated_gb": 0.0}
