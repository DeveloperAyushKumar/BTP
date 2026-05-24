"""ACSA Classifier - MLP for adaptive preset selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PRESET_LABELS = ["eco", "balanced", "quality"]


class ACSAClassifier(nn.Module):
    """
    MLP Classifier for Adaptive Compression Selection.
    
    Input: 768-dim CLIP features + 4 hardware features = 772 dims
    Output: 3 classes (eco, balanced, quality)
    """

    def __init__(self, input_dim: int = 772, hidden_dims: tuple = (128, 64)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], len(PRESET_LABELS)),
        )
        self._feature_mean: Optional[torch.Tensor] = None
        self._feature_std: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict(self, features: torch.Tensor) -> tuple[str, float]:
        """Predict preset with confidence score."""
        self.eval()
        with torch.no_grad():
            if self._feature_mean is not None:
                features = (features - self._feature_mean) / (self._feature_std + 1e-8)
            logits = self.forward(features)
            probs = torch.softmax(logits, dim=-1)
            idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, idx].item()
        return PRESET_LABELS[idx], confidence

    def save(self, path: Path):
        """Save model weights and normalization stats."""
        state = {
            "model_state": self.state_dict(),
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
        }
        torch.save(state, path)
        logger.info(f"ACSA classifier saved to {path}")

    def load(self, path: Path) -> bool:
        """Load model weights. Returns True if successful."""
        if not path.exists():
            logger.warning(f"No trained model at {path}")
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state["model_state"])
        self._feature_mean = state.get("feature_mean")
        self._feature_std = state.get("feature_std")
        logger.info("ACSA classifier loaded")
        return True
