"""Feature Extractor - CLIP-based image embeddings for ACSA."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


class FeatureExtractor:
    """Extracts CLIP image embeddings for ACSA input."""

    def __init__(self):
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None

    def _load_model(self):
        """Load CLIP model (cached after first call)."""
        if self._model is None:
            logger.info(f"Loading CLIP model: {CLIP_MODEL_ID}")
            self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
            self._model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=torch.float32)
            self._model.eval()
            logger.info("CLIP model loaded")

    def extract(self, image: Image.Image) -> torch.Tensor:
        """
        Extract 768-dimensional CLIP embedding from an image.
        
        Returns:
            Tensor of shape (1, 768)
        """
        self._load_model()
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
        # Normalize
        features = outputs / outputs.norm(dim=-1, keepdim=True)
        return features  # (1, 768)
