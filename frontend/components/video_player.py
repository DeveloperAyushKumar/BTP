"""Video Player component."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_video_player(video_path: str | Path):
    """Render video playback with download option."""
    path = Path(video_path)
    if not path.exists():
        st.error(f"Video file not found: {path}")
        return

    st.video(str(path))

    with open(path, "rb") as f:
        st.download_button(
            "📥 Download Video",
            data=f,
            file_name=path.name,
            mime="video/mp4",
        )
