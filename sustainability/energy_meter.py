"""Energy Meter - Hardware-based power measurement for NVIDIA GPUs (RAPL + NVML) with Apple Silicon fallback."""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

# Apple Silicon fallback power profiles (watts)
M3_PRO_IDLE_WATTS = 5.0
M3_PRO_GPU_FULL_WATTS = 22.0
M3_PRO_CPU_FULL_WATTS = 15.0

# RAPL sysfs paths (Intel/AMD CPUs on Linux)
RAPL_BASE = Path("/sys/class/powercap/intel-rapl")


def _nvml_available() -> bool:
    """Check if NVIDIA Management Library is available."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def _rapl_available() -> bool:
    """Check if Intel RAPL counters are readable."""
    energy_file = RAPL_BASE / "intel-rapl:0" / "energy_uj"
    return energy_file.exists() and energy_file.stat().st_size > 0


def _read_rapl_energy_uj() -> float:
    """Read total RAPL energy in microjoules (sum of all packages)."""
    total_uj = 0.0
    try:
        for pkg_dir in sorted(RAPL_BASE.glob("intel-rapl:*")):
            energy_file = pkg_dir / "energy_uj"
            if energy_file.exists():
                total_uj += float(energy_file.read_text().strip())
    except (PermissionError, OSError) as e:
        logger.debug(f"RAPL read failed: {e}")
    return total_uj


@dataclass
class EnergyMeasurement:
    """Result of an energy measurement session."""
    duration_seconds: float
    avg_cpu_percent: float
    avg_memory_percent: float
    estimated_watts: float
    energy_wh: float
    energy_kwh: float
    # Hardware measurement details
    gpu_energy_wh: float = 0.0
    cpu_energy_wh: float = 0.0
    measurement_method: str = "estimation"  # "hardware_nvml_rapl", "hardware_nvml", "estimation"
    gpu_power_samples: int = 0
    avg_gpu_watts: float = 0.0


class EnergyMeter:
    """Measures energy consumption using hardware sensors (NVML + RAPL) or estimation fallback.

    On Kaggle/NVIDIA systems:
      - GPU power: pynvml (reads actual GPU power sensor, e.g., T4 = 0-70W)
      - CPU power: Intel RAPL counters (reads CPU package energy)

    On Apple Silicon (fallback):
      - Estimation based on psutil CPU utilization and known TDP values.
    """

    def __init__(self):
        self._start_time: float = 0.0
        self._cpu_samples: list[float] = []
        self._mem_samples: list[float] = []
        self._gpu_power_samples: list[float] = []  # watts from NVML
        self._running = False

        # Detect available hardware sensors
        self._has_nvml = _nvml_available()
        self._has_rapl = _rapl_available()
        self._nvml_handle = None
        self._rapl_start_uj: float = 0.0

        if self._has_nvml:
            import pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
            logger.info(f"EnergyMeter: NVML available — GPU: {gpu_name}")

        if self._has_rapl:
            logger.info("EnergyMeter: Intel RAPL available for CPU energy")

        if not self._has_nvml and not self._has_rapl:
            logger.info("EnergyMeter: No hardware sensors, using estimation fallback")

    def start(self):
        """Start energy measurement."""
        self._start_time = time.time()
        self._cpu_samples = []
        self._mem_samples = []
        self._gpu_power_samples = []
        self._running = True

        # Record RAPL baseline
        if self._has_rapl:
            self._rapl_start_uj = _read_rapl_energy_uj()

        # Take initial sample
        self._sample()

    def _sample(self):
        """Take a system utilization sample + hardware power reading."""
        self._cpu_samples.append(psutil.cpu_percent(interval=None))
        self._mem_samples.append(psutil.virtual_memory().percent)

        # Read GPU power from NVML (milliwatts → watts)
        if self._has_nvml and self._nvml_handle is not None:
            try:
                import pynvml
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                self._gpu_power_samples.append(power_mw / 1000.0)
            except Exception:
                pass

    def update(self):
        """Call periodically during generation to collect samples."""
        if self._running:
            self._sample()

    def stop(self) -> EnergyMeasurement:
        """Stop measurement and calculate energy consumed."""
        self._running = False
        self._sample()  # Final sample

        duration = time.time() - self._start_time
        avg_cpu = sum(self._cpu_samples) / max(len(self._cpu_samples), 1)
        avg_mem = sum(self._mem_samples) / max(len(self._mem_samples), 1)

        gpu_energy_wh = 0.0
        cpu_energy_wh = 0.0
        avg_gpu_watts = 0.0
        method = "estimation"

        # --- GPU energy from NVML (trapezoidal integration of power samples) ---
        if self._has_nvml and len(self._gpu_power_samples) >= 2:
            avg_gpu_watts = sum(self._gpu_power_samples) / len(self._gpu_power_samples)
            # Energy = avg_power * duration
            gpu_energy_wh = (avg_gpu_watts * duration) / 3600.0
            method = "hardware_nvml"

        # --- CPU energy from RAPL ---
        if self._has_rapl:
            rapl_end_uj = _read_rapl_energy_uj()
            cpu_energy_uj = rapl_end_uj - self._rapl_start_uj
            # Handle counter overflow (RAPL counters wrap around)
            if cpu_energy_uj < 0:
                cpu_energy_uj += 2**32  # 32-bit counter wrap
            cpu_energy_wh = cpu_energy_uj / 1e6 / 3600.0  # uJ → J → Wh
            method = "hardware_nvml_rapl" if self._has_nvml else "hardware_rapl"

        # --- Fallback: estimation for Apple Silicon / no sensors ---
        if not self._has_nvml and not self._has_rapl:
            cpu_watts = M3_PRO_CPU_FULL_WATTS * (avg_cpu / 100.0)
            gpu_watts = M3_PRO_GPU_FULL_WATTS * 0.85
            idle_watts = M3_PRO_IDLE_WATTS
            estimated_total_watts = idle_watts + cpu_watts + gpu_watts
            gpu_energy_wh = (gpu_watts * duration) / 3600.0
            cpu_energy_wh = ((idle_watts + cpu_watts) * duration) / 3600.0
        else:
            estimated_total_watts = avg_gpu_watts + (cpu_energy_wh * 3600.0 / max(duration, 0.1))

        total_energy_wh = gpu_energy_wh + cpu_energy_wh
        total_energy_kwh = total_energy_wh / 1000.0

        measurement = EnergyMeasurement(
            duration_seconds=duration,
            avg_cpu_percent=avg_cpu,
            avg_memory_percent=avg_mem,
            estimated_watts=estimated_total_watts,
            energy_wh=total_energy_wh,
            energy_kwh=total_energy_kwh,
            gpu_energy_wh=gpu_energy_wh,
            cpu_energy_wh=cpu_energy_wh,
            measurement_method=method,
            gpu_power_samples=len(self._gpu_power_samples),
            avg_gpu_watts=avg_gpu_watts,
        )

        logger.info(
            f"Energy measurement [{method}]: {total_energy_wh:.4f} Wh over {duration:.1f}s "
            f"(GPU: {gpu_energy_wh:.4f} Wh @ {avg_gpu_watts:.1f}W avg, "
            f"CPU: {cpu_energy_wh:.4f} Wh, {len(self._gpu_power_samples)} GPU samples)"
        )
        return measurement
