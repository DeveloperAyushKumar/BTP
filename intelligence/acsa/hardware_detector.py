"""Hardware Detector - System specs for ACSA."""

from __future__ import annotations

import logging
import platform

import psutil

logger = logging.getLogger(__name__)


def detect_hardware() -> dict:
    """
    Detect system hardware specifications.
    
    Returns:
        Dictionary with gpu_type, available_vram_gb, total_ram_gb,
        current_cpu_usage_percent, compute_score (0-1).
    """
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)

    # Determine GPU type
    gpu_type = "Unknown"
    chip = platform.processor()
    if "arm" in chip or platform.machine() == "arm64":
        # Apple Silicon
        import subprocess
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True,
            )
            gpu_type = result.stdout.strip() or "Apple Silicon"
        except Exception:
            gpu_type = "Apple Silicon"

    total_ram_gb = mem.total / (1024**3)
    available_vram_gb = mem.available / (1024**3)

    # Compute score: normalized 0-1 based on available memory and CPU
    # M3 Pro with 18GB ~0.7, 36GB ~0.9
    compute_score = min(1.0, available_vram_gb / 20.0) * (1.0 - cpu_percent / 200.0)

    return {
        "gpu_type": gpu_type,
        "available_vram_gb": round(available_vram_gb, 2),
        "total_ram_gb": round(total_ram_gb, 2),
        "current_cpu_usage_percent": round(cpu_percent, 1),
        "compute_score": round(compute_score, 3),
    }
