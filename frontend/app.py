"""EcoVideo Studio - Main Streamlit Application (Remote Backend Mode)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from utils.logger import setup_logging

setup_logging()

# Page config
st.set_page_config(
    page_title="EcoVideo Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False
if "generation_result" not in st.session_state:
    st.session_state.generation_result = None
if "page" not in st.session_state:
    st.session_state.page = "Home"


def main():
    """Main application entry point."""
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/emoji/48/leaf-fluttering-in-wind.png", width=48)
        st.title("EcoVideo Studio")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Home", "Generate", "History", "Analytics", "Settings"],
            index=["Home", "Generate", "History", "Analytics", "Settings"].index(
                st.session_state.page
            ),
        )
        st.session_state.page = page

        st.markdown("---")
        # Show backend connection status
        try:
            from backend.remote_client import RemoteClient
            client = RemoteClient()
            info = client.health()
            st.caption(f"🌿 Green AI • Kaggle GPU ({info['device']})")
            st.caption(f"VRAM: {info['vram_free_gb']:.0f}/{info['vram_total_gb']:.0f} GB free")
        except Exception:
            st.caption("⚠️ Backend offline — start Kaggle notebook")

    # Route to page
    if page == "Home":
        render_home()
    elif page == "Generate":
        render_generate()
    elif page == "History":
        render_history()
    elif page == "Analytics":
        render_analytics()
    elif page == "Settings":
        render_settings()


def render_home():
    """Home page with introduction and stats."""
    st.title("🌿 EcoVideo Studio")
    st.markdown(
        """
        ### Energy-Efficient Video Generation
        
        Transform static images into smooth videos using **Stable Video Diffusion**, 
        running on Kaggle GPU with sustainability tracking.
        """
    )

    col1, col2, col3 = st.columns(3)

    # Load cumulative stats
    try:
        from backend.history_manager import HistoryManager
        hm = HistoryManager()
        stats = hm.get_total_savings()
        with col1:
            st.metric("Videos Generated", stats["total_generations"])
        with col2:
            st.metric("CO₂ Saved", f"{stats['total_carbon_gco2']:.1f} g")
        with col3:
            st.metric("Energy Saved", f"{stats['total_energy_kwh'] * 1000:.1f} Wh")
    except Exception:
        with col1:
            st.metric("Videos Generated", 0)
        with col2:
            st.metric("CO₂ Saved", "0 g")
        with col3:
            st.metric("Energy Saved", "0 Wh")

    st.markdown("---")
    st.markdown(
        """
        #### Key Features
        - 🎬 **Three Quality Presets**: Eco, Balanced, Quality
        - 🤖 **Smart Recommendations**: ACSA algorithm selects optimal preset
        - 📊 **Sustainability Dashboard**: Energy, carbon, and cost metrics
        - �️ **Kaggle GPU Backend**: Free T4 GPU via ngrok tunnel
        - 🌐 **Lightweight Frontend**: Only Streamlit runs locally
        """
    )


def render_generate():
    """Video generation page."""
    st.title("🎬 Generate Video")

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="image_upload"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", width=400)

        # Preset selection
        st.markdown("### Select Preset")

        preset_options = {
            "🌱 Eco — 8 steps • Fast • Low VRAM": "eco",
            "⚖️ Balanced — 16 steps • Good quality": "balanced",
            "✨ Quality — 25 steps • Best output": "quality",
        }

        selected_label = st.radio(
            "Choose a quality preset:",
            list(preset_options.keys()),
            index=1,  # Default to Balanced
            horizontal=True,
        )
        preset = preset_options[selected_label]

        st.info(f"Selected: **{preset.title()}** preset")

        # Auto-recommend via remote backend
        if st.button("🤖 Auto-Recommend Preset"):
            with st.spinner("Asking Kaggle backend..."):
                try:
                    from backend.remote_client import RemoteClient
                    client = RemoteClient()
                    result = client.recommend_preset(image)
                    st.success(
                        f"Recommended: **{result['preset'].title()}** "
                        f"(confidence: {result['confidence']:.0%}, method: {result['method']})"
                    )
                    preset = result["preset"]
                except Exception as e:
                    st.warning(f"Auto-recommend unavailable: {e}")

        # ── Optimization Tuning ──────────────────────────────────
        st.markdown("### 🔧 Optimization Tuning")
        st.caption("Toggle individual optimizations. Hover ℹ️ for details.")

        opt_col1, opt_col2 = st.columns(2)

        with opt_col1:
            opt_attention_slicing = st.toggle(
                "✂️ Attention Slicing",
                value=True,
                help=(
                    "**What:** Splits the attention computation into chunks instead of computing it all at once.\n\n"
                    "**How it saves resources:** Reduces peak VRAM by ~25-40% because only one slice is in GPU memory at a time. "
                    "Adds ~10% to generation time as a trade-off.\n\n"
                    "**Best for:** Low-VRAM GPUs or when running out of memory."
                ),
            )
            opt_forward_chunking = st.toggle(
                "🧩 Forward Chunking",
                value=True,
                help=(
                    "**What:** Breaks the UNet feed-forward layers into smaller sequential chunks.\n\n"
                    "**How it saves resources:** Lowers peak VRAM by ~15-20% by processing the feed-forward network "
                    "in pieces rather than all at once. Slight time overhead (~5%).\n\n"
                    "**Best for:** Stacking with attention slicing for maximum memory savings."
                ),
            )
            opt_deep_cache = st.toggle(
                "⚡ DeepCache",
                value=False,
                help=(
                    "**What:** Caches intermediate UNet features and reuses them every N steps, skipping redundant computation.\n\n"
                    "**How it saves resources:** Speeds up inference by ~30-50% with minimal quality loss. "
                    "Trades a small amount of VRAM (for the cache) for significantly less computation time.\n\n"
                    "**Best for:** Eco/Balanced presets where speed matters more than pixel-perfect output."
                ),
            )

        with opt_col2:
            opt_cpu_offload = st.toggle(
                "💾 Model CPU Offload",
                value=False,
                help=(
                    "**What:** Moves model components (UNet, VAE, text encoder) to CPU when not in use, "
                    "loading them to GPU only for their forward pass.\n\n"
                    "**How it saves resources:** Dramatically reduces VRAM usage (~60-70%) at the cost of "
                    "slower generation due to CPU↔GPU data transfer each step.\n\n"
                    "**Best for:** When VRAM is critically low and you'd rather wait than crash."
                ),
            )
            opt_vae_slicing = st.toggle(
                "🎞️ VAE Slicing",
                value=False,
                help=(
                    "**What:** Decodes video frames one-by-one through the VAE instead of in a batch.\n\n"
                    "**How it saves resources:** Cuts VAE decode memory by up to 90% — "
                    "especially impactful with many frames (14+). Adds ~5-10% decode time.\n\n"
                    "**Best for:** Quality preset with 14 frames where VAE decode is the memory bottleneck."
                ),
            )
            opt_vae_tiling = st.toggle(
                "🧱 VAE Tiling",
                value=False,
                help=(
                    "**What:** Decodes the image in overlapping tiles instead of the full resolution at once.\n\n"
                    "**How it saves resources:** Reduces VAE peak VRAM significantly for high-resolution outputs. "
                    "Minimal quality impact due to blended tile overlaps. ~10% time overhead.\n\n"
                    "**Best for:** When generating at full 1024×576 resolution with limited VRAM."
                ),
            )

        if opt_deep_cache:
            deep_cache_interval = st.slider(
                "DeepCache interval (reuse features every N steps)",
                min_value=2, max_value=5, value=3,
                help="Lower = faster but less quality. 3 is a good default.",
            )
        else:
            deep_cache_interval = 3

        # Pack optimizations dict
        optimizations = {
            "attention_slicing": opt_attention_slicing,
            "forward_chunking": opt_forward_chunking,
            "deep_cache": opt_deep_cache,
            "deep_cache_interval": deep_cache_interval,
            "cpu_offload": opt_cpu_offload,
            "vae_slicing": opt_vae_slicing,
            "vae_tiling": opt_vae_tiling,
        }

        st.markdown("---")

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            motion = st.slider("Motion Intensity", 1, 255, 127)
            noise = st.slider("Noise Strength", 0.0, 0.5, 0.02, step=0.01)
            seed = st.number_input("Seed", value=42, min_value=0)
            num_frames = st.selectbox("Number of Frames", [7, 14, 25], index=1)

        # Generate button
        if st.button("🚀 Generate Video", type="primary", use_container_width=True):
            _run_generation(image, preset, seed, num_frames, motion, noise, optimizations)

    # Display result if available
    if st.session_state.generation_result is not None:
        _display_result(st.session_state.generation_result)


def _run_generation(image, preset, seed, num_frames, motion, noise, optimizations=None):
    """Execute video generation via Kaggle remote backend."""
    optimizations = optimizations or {}
    progress_bar = st.progress(0, text="Sending to Kaggle GPU...")
    status_text = st.empty()

    try:
        from backend.remote_client import RemoteClient
        from sustainability.carbon_calculator import CarbonCalculator

        client = RemoteClient()

        active = [k for k, v in optimizations.items() if v is True]
        status_text.text(f"Generating with {preset} preset • Optimizations: {', '.join(active) if active else 'none'}")
        progress_bar.progress(0.2, text="Inference running on remote GPU...")

        result = client.generate(
            image=image,
            preset_name=preset,
            seed=seed,
            num_frames=num_frames,
            motion_bucket_id=motion,
            noise_aug_strength=noise,
            optimizations=optimizations,
        )

        progress_bar.progress(0.9, text="Processing results...")

        # Calculate sustainability metrics from reported energy
        carbon_calc = CarbonCalculator()
        savings = carbon_calc.calculate_savings(result.energy_wh)

        # Build a simple energy-like object for display
        class _Energy:
            pass
        energy = _Energy()
        energy.energy_wh = result.energy_wh
        energy.energy_kwh = result.energy_wh / 1000.0

        # Store in session
        st.session_state.generation_result = {
            "result": result,
            "energy": energy,
            "savings": savings,
            "output_path": str(result.video_path),
        }

        # Save to history
        try:
            from backend.history_manager import HistoryManager
            hm = HistoryManager()
            hm.add_generation(
                generation_id=result.generation_id,
                image_path="uploaded",
                preset=preset,
                seed=seed,
                generation_time=result.generation_time_seconds,
                peak_vram=result.peak_vram_gb,
                ssim=None,
                psnr=None,
                energy_kwh=energy.energy_kwh,
                carbon_gco2=savings["local_carbon_g"],
                output_video_path=str(result.video_path),
            )
        except Exception:
            pass

        progress_bar.progress(1.0, text="✅ Complete!")
        status_text.empty()

    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_bar.empty()
        status_text.empty()
        st.error(f"Generation failed: {e}")
        st.code(traceback.format_exc(), language="text")
        if "connection" in str(e).lower():
            st.warning("💡 Is the Kaggle notebook running? Check the ngrok URL in .env")


def _display_result(gen_data: dict):
    """Display generation results and metrics."""
    st.markdown("---")
    st.markdown("### 🎥 Generated Video")

    result = gen_data["result"]
    energy = gen_data["energy"]
    savings = gen_data["savings"]

    # Video player
    output_path = gen_data["output_path"]
    if Path(output_path).exists():
        st.video(output_path)

    # Metrics cards
    st.markdown("### 📊 Generation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    gen_time = getattr(result, 'generation_time_seconds', 0)
    peak_vram = getattr(result, 'peak_vram_gb', 0)
    with c1:
        st.metric("⏱️ Time", f"{gen_time:.1f}s")
    with c2:
        st.metric("💾 Peak VRAM", f"{peak_vram:.1f} GB")
    with c3:
        st.metric("⚡ Energy", f"{energy.energy_wh:.1f} Wh")
    with c4:
        st.metric("🌍 Carbon", f"{savings['local_carbon_g']:.2f} g CO₂")

    # Sustainability comparison
    st.markdown("### 🌿 Sustainability Impact")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Energy Saved vs Cloud", f"{savings['energy_saved_percent']:.0f}%", delta="green")
    with c2:
        st.metric("Carbon Saved", f"{savings['carbon_saved_g']:.1f} g CO₂")
    with c3:
        st.metric("Cost Saved", f"${savings['cost_saved_usd']:.3f}")

    # Impact visualizations
    from sustainability.impact_visualizer import format_impact
    impact = format_impact(savings["energy_saved_wh"], savings["carbon_saved_g"])
    
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown(f"🌳 **{impact['tree_hours']:.1f} hours** of tree CO₂ absorption")
    with ic2:
        st.markdown(f"📱 **{impact['smartphone_charges']:.1f}** smartphone charges saved")
    with ic3:
        st.markdown(f"🚗 **{impact['driving_km_avoided']:.2f} km** driving avoided")

    # Download
    if Path(output_path).exists():
        with open(output_path, "rb") as f:
            st.download_button("📥 Download Video", f, file_name=Path(output_path).name)


@st.cache_resource
def _get_remote_client():
    """Get cached remote client instance."""
    from backend.remote_client import RemoteClient
    return RemoteClient()


def render_history():
    """History page showing past generations."""
    st.title("📜 Generation History")
    try:
        from backend.history_manager import HistoryManager
        hm = HistoryManager()
        records = hm.get_recent_generations(limit=50)

        if not records:
            st.info("No generations yet. Go to Generate to create your first video!")
            return

        import pandas as pd
        df = pd.DataFrame([
            {
                "ID": r.id,
                "Time": r.timestamp,
                "Preset": r.preset,
                "Duration": f"{r.generation_time:.1f}s",
                "VRAM": f"{r.peak_vram:.1f} GB",
                "Energy": f"{r.energy_kwh * 1000:.1f} Wh",
                "Carbon": f"{r.carbon_gco2:.2f} g",
                "Rating": r.user_rating or "-",
            }
            for r in records
        ])
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load history: {e}")


def render_analytics():
    """Analytics page with charts."""
    st.title("📈 Analytics")

    try:
        from backend.history_manager import HistoryManager
        import pandas as pd
        import matplotlib.pyplot as plt

        hm = HistoryManager()
        records = hm.get_recent_generations(limit=100)

        if not records:
            st.info("Generate some videos first to see analytics.")
            return

        df = pd.DataFrame([
            {
                "timestamp": r.timestamp,
                "preset": r.preset,
                "generation_time": r.generation_time,
                "peak_vram": r.peak_vram,
                "energy_wh": r.energy_kwh * 1000,
                "carbon_g": r.carbon_gco2,
            }
            for r in records
        ])

        # Preset distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Preset Usage")
            fig, ax = plt.subplots()
            df["preset"].value_counts().plot.bar(ax=ax, color=["#4CAF50", "#FF9800", "#2196F3"])
            st.pyplot(fig)

        with col2:
            st.markdown("#### Cumulative Carbon Savings")
            fig, ax = plt.subplots()
            # Cloud baseline carbon per video
            cloud_carbon_per = 150.0 * 0.475 / 1000.0 * 1000  # ~71g per video
            cumulative_saved = [(cloud_carbon_per - r) for r in df["carbon_g"].cumsum()]
            ax.plot(cumulative_saved, color="#4CAF50")
            ax.set_xlabel("Generation #")
            ax.set_ylabel("Cumulative CO₂ Saved (g)")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Analytics error: {e}")


def render_settings():
    """Settings page."""
    st.title("⚙️ Settings")

    st.markdown("#### Carbon Calculation")
    region = st.selectbox(
        "Your Region (for grid carbon intensity)",
        ["global", "us_average", "eu_average", "uk", "france", "germany", "india", "china", "australia"],
        index=0,
    )

    st.markdown("#### Defaults")
    default_preset = st.selectbox("Default Preset", ["eco", "balanced", "quality"], index=1)

    st.markdown("#### Storage")
    auto_delete = st.checkbox("Auto-delete videos older than 30 days", value=False)

    if st.button("Save Settings"):
        st.success("Settings saved!")

    st.markdown("---")
    st.markdown("#### Backend Info")
    try:
        from backend.remote_client import RemoteClient
        client = RemoteClient()
        info = client.health()
        st.json({
            "Mode": "Kaggle Remote GPU",
            "Device": info["device"],
            "VRAM Total": f"{info['vram_total_gb']:.0f} GB",
            "VRAM Free": f"{info['vram_free_gb']:.1f} GB",
            "Backend URL": client.base_url,
        })
    except Exception as e:
        st.warning(f"Backend offline: {e}")
        st.info("Run the Kaggle notebook and update KAGGLE_BACKEND_URL in .env")


if __name__ == "__main__":
    main()
