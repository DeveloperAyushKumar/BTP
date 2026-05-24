"""Metrics Dashboard component."""

from __future__ import annotations

import streamlit as st

from sustainability.impact_visualizer import format_impact


def render_dashboard(generation_data: dict):
    """
    Render the sustainability metrics dashboard.
    
    Args:
        generation_data: Dict with 'result', 'energy', 'savings' keys
    """
    result = generation_data["result"]
    energy = generation_data["energy"]
    savings = generation_data["savings"]

    st.markdown("### 📊 Sustainability Dashboard")

    # 2x2 metric cards
    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2 = st.columns(2)

    with row1_c1:
        st.metric(
            "⚡ Energy Used",
            f"{energy.energy_wh:.2f} Wh",
            delta=f"-{savings['energy_saved_percent']:.0f}% vs cloud",
            delta_color="inverse",
        )

    with row1_c2:
        carbon_color = "normal" if savings["local_carbon_g"] < 5 else "off"
        st.metric(
            "🌍 Carbon Footprint",
            f"{savings['local_carbon_g']:.2f} g CO₂",
            delta=f"-{savings['carbon_saved_g']:.1f}g saved",
            delta_color="inverse",
        )

    with row2_c1:
        st.metric(
            "☁️ Cloud Savings",
            f"{savings['energy_saved_percent']:.0f}%",
            help="Energy saved compared to cloud generation",
        )

    with row2_c2:
        # Time comparison (cloud ~30s but with queue/overhead)
        st.metric(
            "💰 Cost Saved",
            f"${savings['cost_saved_usd']:.3f}",
            help="Estimated API cost avoided",
        )

    # Impact visualization
    st.markdown("---")
    st.markdown("#### 🌿 Environmental Impact")

    impact = format_impact(savings["energy_saved_wh"], savings["carbon_saved_g"])

    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.markdown(f"🌳\n\n**{impact['tree_hours']:.1f}h**\n\nTree absorption")
    with ic2:
        st.markdown(f"📱\n\n**{impact['smartphone_charges']:.1f}**\n\nPhone charges saved")
    with ic3:
        st.markdown(f"🚗\n\n**{impact['driving_km_avoided']:.3f} km**\n\nDriving avoided")
    with ic4:
        st.markdown(f"💡\n\n**{impact['led_bulb_hours']:.1f}h**\n\nLED bulb hours")
