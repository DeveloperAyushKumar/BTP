"""Cloud Comparator - Compare local vs cloud generation costs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Cloud pricing estimates (USD)
CLOUD_PROVIDERS = {
    "replicate": {"cost_per_video": 0.05, "avg_time_seconds": 30},
    "runpod": {"cost_per_video": 0.04, "avg_time_seconds": 25},
    "aws_sagemaker": {"cost_per_video": 0.08, "avg_time_seconds": 20},
}


@dataclass
class CloudComparison:
    """Comparison between local and cloud generation."""
    local_time_seconds: float
    local_energy_wh: float
    local_cost_usd: float  # Electricity cost
    cloud_cost_usd: float
    cloud_energy_wh: float
    cumulative_savings_usd: float
    videos_until_roi: int  # Not applicable for free local


class CloudComparator:
    """Compares local generation against cloud alternatives."""

    ELECTRICITY_RATE_USD_PER_KWH = 0.15  # Average US rate

    def compare(
        self,
        local_energy_wh: float,
        local_time_seconds: float,
        total_generations: int = 1,
    ) -> CloudComparison:
        """Compare a local generation against cloud baseline."""
        # Local electricity cost
        local_cost = (local_energy_wh / 1000.0) * self.ELECTRICITY_RATE_USD_PER_KWH

        # Average cloud cost
        avg_cloud_cost = sum(p["cost_per_video"] for p in CLOUD_PROVIDERS.values()) / len(CLOUD_PROVIDERS)
        cloud_energy_wh = 150.0  # Baseline from research

        # Cumulative savings
        cumulative_savings = (avg_cloud_cost - local_cost) * total_generations

        return CloudComparison(
            local_time_seconds=local_time_seconds,
            local_energy_wh=local_energy_wh,
            local_cost_usd=round(local_cost, 4),
            cloud_cost_usd=round(avg_cloud_cost, 4),
            cloud_energy_wh=cloud_energy_wh,
            cumulative_savings_usd=round(cumulative_savings, 2),
            videos_until_roi=0,  # Already saving from video 1
        )
