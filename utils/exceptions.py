"""Custom exceptions for EcoVideo Studio."""

from __future__ import annotations


class EcoVideoError(Exception):
    """Base exception for EcoVideo Studio."""
    pass


class ModelLoadError(EcoVideoError):
    """Failed to load the SVD model."""
    pass


class GenerationError(EcoVideoError):
    """Video generation failed."""
    pass


class MemoryError(EcoVideoError):
    """Insufficient memory for operation."""
    pass


class InvalidImageError(EcoVideoError):
    """Input image is invalid or unsupported."""
    pass


class DatabaseError(EcoVideoError):
    """Database operation failed."""
    pass
