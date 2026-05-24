"""Preset Selector component."""

from __future__ import annotations

import streamlit as st

from backend.config_manager import PRESETS


def render_preset_selector(default: str = "balanced") -> str:
    """Render preset selection cards and return chosen preset name."""
    st.markdown("### Choose Quality Preset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🌱 Eco")
        st.caption("8 steps • 384×256 • ~90-150s")
        st.caption("Lowest energy & VRAM usage")
        eco = st.button("Select Eco", use_container_width=True)

    with col2:
        st.markdown("#### ⚖️ Balanced")
        st.caption("16 steps • 512×320 • ~180-240s")
        st.caption("Good quality/efficiency tradeoff")
        bal = st.button("Select Balanced", use_container_width=True)

    with col3:
        st.markdown("#### ✨ Quality")
        st.caption("25 steps • 512×320 • ~300-420s")
        st.caption("Best visual output")
        qual = st.button("Select Quality", use_container_width=True)

    if eco:
        return "eco"
    elif qual:
        return "quality"
    elif bal:
        return "balanced"
    return default
