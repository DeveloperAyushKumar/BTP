"""Progress Display component."""

from __future__ import annotations

import streamlit as st

from utils.helpers import format_time


def render_progress(step: int, total_steps: int, elapsed: float, preset: str):
    """Render generation progress bar with ETA."""
    progress = step / total_steps if total_steps > 0 else 0

    # Estimate remaining time
    if step > 0:
        time_per_step = elapsed / step
        remaining = time_per_step * (total_steps - step)
        eta_text = f"ETA: {format_time(remaining)}"
    else:
        eta_text = "Starting..."

    st.progress(progress, text=f"Step {step}/{total_steps} • {format_time(elapsed)} elapsed • {eta_text}")
