"""Test Sustainability - Verify energy and carbon calculations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class TestEnergyMeter:
    def test_measurement(self):
        import time
        from sustainability.energy_meter import EnergyMeter

        meter = EnergyMeter()
        meter.start()
        time.sleep(0.1)
        result = meter.stop()

        assert result.duration_seconds >= 0.1
        assert result.energy_wh > 0
        assert result.energy_kwh > 0
        assert result.estimated_watts > 0

    def test_energy_reasonable(self):
        """Energy for ~2min generation should be 5-20 Wh."""
        import time
        from sustainability.energy_meter import EnergyMeter

        meter = EnergyMeter()
        meter.start()
        time.sleep(0.05)
        result = meter.stop()

        # Extrapolate to 2 minutes
        rate_wh_per_sec = result.energy_wh / result.duration_seconds
        two_min_energy = rate_wh_per_sec * 120
        # Should be roughly 5-20 Wh for M3 Pro
        assert 1 < two_min_energy < 50


class TestCarbonCalculator:
    def test_emissions(self):
        from sustainability.carbon_calculator import CarbonCalculator
        calc = CarbonCalculator(region="global")
        # 0.01 kWh * 475 gCO2/kWh = 4.75g
        emissions = calc.calculate_emissions(0.01)
        assert abs(emissions - 4.75) < 0.01

    def test_savings(self):
        from sustainability.carbon_calculator import CarbonCalculator
        calc = CarbonCalculator()
        savings = calc.calculate_savings(local_energy_wh=10.0)
        assert savings["energy_saved_wh"] > 0
        assert savings["carbon_saved_g"] > 0
        assert savings["energy_saved_percent"] > 80  # Should be >90%


class TestImpactVisualizer:
    def test_format_impact(self):
        from sustainability.impact_visualizer import format_impact
        impact = format_impact(energy_saved_wh=140.0, carbon_saved_g=60.0)
        assert impact["smartphone_charges"] > 0
        assert impact["tree_hours"] > 0
        assert impact["driving_km_avoided"] > 0


class TestCloudComparator:
    def test_compare(self):
        from sustainability.cloud_comparator import CloudComparator
        comp = CloudComparator()
        result = comp.compare(local_energy_wh=10.0, local_time_seconds=120.0, total_generations=5)
        assert result.cloud_cost_usd > result.local_cost_usd
        assert result.cumulative_savings_usd > 0
