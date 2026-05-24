"""Quality Validator - SSIM, PSNR, and temporal consistency metrics."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logger = logging.getLogger(__name__)


class QualityValidator:
    """Calculates video quality metrics."""

    @staticmethod
    def calculate_ssim(frames_a: list[Image.Image], frames_b: list[Image.Image]) -> float:
        """Calculate average SSIM between two frame sequences."""
        scores = []
        for fa, fb in zip(frames_a, frames_b):
            a = np.array(fa.convert("L"))
            b = np.array(fb.convert("L"))
            ssim_val = structural_similarity(a, b, data_range=255)
            scores.append(ssim_val)
        return float(np.mean(scores))

    @staticmethod
    def calculate_psnr(frames_a: list[Image.Image], frames_b: list[Image.Image]) -> float:
        """Calculate average PSNR between two frame sequences."""
        scores = []
        for fa, fb in zip(frames_a, frames_b):
            a = np.array(fa).astype(np.float64)
            b = np.array(fb).astype(np.float64)
            if np.array_equal(a, b):
                scores.append(100.0)  # Cap for identical frames
            else:
                psnr_val = peak_signal_noise_ratio(a, b, data_range=255)
                scores.append(min(psnr_val, 100.0))
        return float(np.mean(scores))

    @staticmethod
    def calculate_temporal_consistency(frames: list[Image.Image]) -> float:
        """
        Measure motion smoothness using optical flow.
        Lower values indicate smoother motion.
        """
        flow_magnitudes = []
        for i in range(len(frames) - 1):
            prev = cv2.cvtColor(np.array(frames[i]), cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(np.array(frames[i + 1]), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_magnitudes.append(np.mean(magnitude))

        return float(np.std(flow_magnitudes))

    @staticmethod
    def quick_quality_check(frames: list[Image.Image]) -> dict:
        """Run a quick quality assessment on generated frames."""
        # Compare consecutive frames for temporal quality
        if len(frames) < 2:
            return {"score": 0.0, "warning": "Too few frames"}

        ssim_scores = []
        for i in range(len(frames) - 1):
            a = np.array(frames[i].convert("L"))
            b = np.array(frames[i + 1].convert("L"))
            ssim_val = structural_similarity(a, b, data_range=255)
            ssim_scores.append(ssim_val)

        avg_ssim = float(np.mean(ssim_scores))
        temporal = QualityValidator.calculate_temporal_consistency(frames)

        warning = None
        if avg_ssim < 0.85:
            warning = "Low frame coherence detected. Consider using a higher quality preset."

        return {
            "avg_frame_ssim": avg_ssim,
            "temporal_consistency": temporal,
            "score": avg_ssim,
            "warning": warning,
        }
