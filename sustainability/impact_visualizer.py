"""Impact Visualizer - Human-friendly sustainability metrics."""

from __future__ import annotations


# Constants for conversions
CO2_PER_TREE_YEAR_G = 22000.0  # 22 kg CO2 absorbed per tree per year
SMARTPHONE_CHARGE_WH = 19.0  # ~19 Wh per full charge
CAR_EMISSIONS_G_PER_KM = 251.0  # Average car gCO2/km
LED_BULB_WATTS = 10.0  # 10W LED bulb


def trees_equivalent(carbon_saved_g: float) -> float:
    """Convert CO2 saved to equivalent tree-hours of absorption."""
    tree_years = carbon_saved_g / CO2_PER_TREE_YEAR_G
    tree_hours = tree_years * 365 * 24
    return round(tree_hours, 2)


def smartphone_charges(energy_saved_wh: float) -> float:
    """Convert energy saved to smartphone charges equivalent."""
    return round(energy_saved_wh / SMARTPHONE_CHARGE_WH, 2)


def driving_distance_km(carbon_saved_g: float) -> float:
    """Convert CO2 saved to equivalent driving distance avoided."""
    return round(carbon_saved_g / CAR_EMISSIONS_G_PER_KM, 3)


def led_bulb_hours(energy_saved_wh: float) -> float:
    """Convert energy saved to LED bulb hours."""
    return round(energy_saved_wh / LED_BULB_WATTS, 1)


def format_impact(energy_saved_wh: float, carbon_saved_g: float) -> dict:
    """Get all impact comparisons as a dictionary."""
    return {
        "tree_hours": trees_equivalent(carbon_saved_g),
        "smartphone_charges": smartphone_charges(energy_saved_wh),
        "driving_km_avoided": driving_distance_km(carbon_saved_g),
        "led_bulb_hours": led_bulb_hours(energy_saved_wh),
    }
