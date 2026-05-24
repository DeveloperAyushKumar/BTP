"""Remote Client - Connects local frontend to Kaggle GPU backend."""

from __future__ import annotations

import io
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Load backend URL from environment or .env file
def _get_backend_url() -> str:
    url = os.environ.get("KAGGLE_BACKEND_URL", "")
    if not url:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("KAGGLE_BACKEND_URL="):
                    url = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return url.rstrip("/")


BACKEND_URL = _get_backend_url()


@dataclass
class RemoteGenerationResult:
    """Result returned from Kaggle backend."""
    generation_id: str
    generation_time_seconds: float
    peak_vram_gb: float
    preset_name: str
    num_steps: int
    seed: int
    energy_wh: float
    video_path: Path


class RemoteClient:
    """Client to communicate with the Kaggle FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or BACKEND_URL
        if not self.base_url:
            raise RuntimeError(
                "KAGGLE_BACKEND_URL not set. "
                "Run the Kaggle notebook and paste the ngrok URL into .env"
            )

    def health(self) -> dict:
        """Check if backend is alive."""
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def generate(
        self,
        image: Image.Image,
        preset_name: str = "balanced",
        seed: int = 42,
        num_frames: int = 14,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        output_dir: Optional[Path] = None,
        optimizations: Optional[dict] = None,
    ) -> RemoteGenerationResult:
        """
        Send image to Kaggle backend, receive generated video.
        """
        import json as _json

        # Encode image to bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        data = {
            "preset_name": preset_name,
            "seed": str(seed),
            "num_frames": str(num_frames),
            "motion_bucket_id": str(motion_bucket_id),
            "noise_aug_strength": str(noise_aug_strength),
        }
        if optimizations:
            data["optimizations_json"] = _json.dumps(optimizations)

        resp = requests.post(
            f"{self.base_url}/generate",
            files={"image": ("input.png", buf, "image/png")},
            data=data,
            timeout=300,  # 5 min timeout for generation
        )
        resp.raise_for_status()

        # Parse metadata from headers
        headers = resp.headers
        gen_id = headers.get("X-Generation-Id", "unknown")
        gen_time = float(headers.get("X-Generation-Time", "0"))
        peak_vram = float(headers.get("X-Peak-Vram-Gb", "0"))
        energy_wh = float(headers.get("X-Energy-Wh", "0"))
        preset = headers.get("X-Preset", preset_name)
        steps = int(headers.get("X-Num-Steps", "0"))
        actual_seed = int(headers.get("X-Seed", str(seed)))

        # Save video locally
        output_dir = output_dir or Path(__file__).resolve().parent.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"ecovideo_{gen_id}_{preset}.mp4"
        video_path = output_dir / filename
        video_path.write_bytes(resp.content)

        logger.info(f"Video received: {video_path} ({gen_time:.1f}s, {peak_vram:.1f}GB VRAM)")

        return RemoteGenerationResult(
            generation_id=gen_id,
            generation_time_seconds=gen_time,
            peak_vram_gb=peak_vram,
            preset_name=preset,
            num_steps=steps,
            seed=actual_seed,
            energy_wh=energy_wh,
            video_path=video_path,
        )

    def recommend_preset(self, image: Image.Image) -> dict:
        """Get preset recommendation from backend."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        resp = requests.post(
            f"{self.base_url}/recommend",
            files={"image": ("input.png", buf, "image/png")},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
