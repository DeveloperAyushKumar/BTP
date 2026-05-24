"""Upload Zone component."""

from __future__ import annotations

import streamlit as st
from PIL import Image

from utils.helpers import validate_image, ensure_rgb


def render_upload_zone() -> Image.Image | None:
    """Render the image upload zone and return the uploaded image."""
    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG and PNG images up to 4096x4096",
    )

    if uploaded is None:
        return None

    image = Image.open(uploaded)
    valid, error = validate_image(image)
    if not valid:
        st.error(f"Invalid image: {error}")
        return None

    image = ensure_rgb(image)
    st.image(image, caption=f"Uploaded: {uploaded.name} ({image.size[0]}x{image.size[1]})", width=400)
    return image
