"""Carbon Calculator - CO2 emissions from energy consumption."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CARBON_FACTORS_PATH = DATA_DIR / "grid_carbon_factors.json"

# Cloud baseline: ~150 Wh per video (100W server * 1.5 PUE * ~1 min effective)
CLOUD_BASELINE_WH = 150.0
CLOUD_COST_PER_VIDEO_USD = 0.05


class CarbonCalculator:
    """Calculates carbon emissions and savings."""

    def __init__(self, region: str = "global"):
        self.region = region
        self._factors = self._load_factors()
        self.carbon_intensity = self._factors.get(region, 475.0)  # gCO2/kWh

    def _load_factors(self) -> dict:
        """Load regional carbon intensity factors."""
        if CARBON_FACTORS_PATH.exists():
            with open(CARBON_FACTORS_PATH) as f:
                return json.load(f)
        return {"global": 475.0}

    def calculate_emissions(self, energy_kwh: float) -> float:
        """Calculate CO2 emissions in grams."""
        return energy_kwh * self.carbon_intensity

    def calculate_savings(self, local_energy_wh: float) -> dict:
        """Compare local vs cloud energy and emissions."""
        local_kwh = local_energy_wh / 1000.0
        cloud_kwh = CLOUD_BASELINE_WH / 1000.0

        local_carbon = self.calculate_emissions(local_kwh)
        cloud_carbon = self.calculate_emissions(cloud_kwh)

        energy_saved_wh = CLOUD_BASELINE_WH - local_energy_wh
        carbon_saved_g = cloud_carbon - local_carbon
        cost_saved_usd = CLOUD_COST_PER_VIDEO_USD

        return {
            "local_energy_wh": local_energy_wh,
            "cloud_energy_wh": CLOUD_BASELINE_WH,
            "energy_saved_wh": energy_saved_wh,
            "energy_saved_percent": (energy_saved_wh / CLOUD_BASELINE_WH) * 100,
            "local_carbon_g": local_carbon,
            "cloud_carbon_g": cloud_carbon,
            "carbon_saved_g": carbon_saved_g,
            "cost_saved_usd": cost_saved_usd,
        }
