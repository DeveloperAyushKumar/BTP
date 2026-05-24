"""History Panel component."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from backend.history_manager import HistoryManager


def render_history_panel(limit: int = 20):
    """Render the generation history table."""
    hm = HistoryManager()
    records = hm.get_recent_generations(limit=limit)

    if not records:
        st.info("No generation history yet.")
        return

    df = pd.DataFrame([
        {
            "ID": r.id,
            "Date": r.timestamp[:16],
            "Preset": r.preset.title(),
            "Time": f"{r.generation_time:.0f}s",
            "VRAM": f"{r.peak_vram:.1f}GB",
            "Energy": f"{r.energy_kwh * 1000:.1f}Wh",
            "CO₂": f"{r.carbon_gco2:.2f}g",
            "⭐": r.user_rating or "-",
        }
        for r in records
    ])

    st.dataframe(df, use_container_width=True, hide_index=True)
