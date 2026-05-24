"""Deep Cache - Feature caching for faster inference."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class DeepCache:
    """
    Implements feature caching to skip redundant computations.
    
    Caches intermediate UNet features at specified intervals,
    reusing them for subsequent steps to reduce computation.
    """

    def __init__(self, cache_interval: int = 3):
        self.cache_interval = cache_interval
        self._cache: dict[str, torch.Tensor] = {}
        self._step_count = 0

    def should_cache(self, step: int) -> bool:
        """Determine if this step should compute and cache features."""
        return step % self.cache_interval == 0

    def should_use_cache(self, step: int) -> bool:
        """Determine if this step should use cached features."""
        return not self.should_cache(step) and len(self._cache) > 0

    def store(self, key: str, tensor: torch.Tensor):
        """Store a tensor in the cache."""
        self._cache[key] = tensor

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a cached tensor."""
        return self._cache.get(key)

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._step_count = 0

    @property
    def memory_usage_mb(self) -> float:
        """Estimate memory used by cache in MB."""
        total = sum(t.element_size() * t.nelement() for t in self._cache.values())
        return total / (1024 * 1024)
