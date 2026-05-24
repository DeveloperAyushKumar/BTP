"""Setup App - First-run initialization script."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run first-time setup."""
    logger.info("🌿 EcoVideo Studio - First Time Setup")
    logger.info("=" * 50)

    # 1. Create directories
    logger.info("Creating directories...")
    dirs = [
        PROJECT_ROOT / "data" / "trained_models",
        PROJECT_ROOT / "data" / "benchmarks",
        PROJECT_ROOT / "data" / "test_images",
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("✅ Directories created")

    # 2. Initialize database
    logger.info("Initializing database...")
    from storage.database import initialize_database
    initialize_database()
    logger.info("✅ Database initialized")

    # 3. Verify environment
    logger.info("Verifying environment...")
    try:
        import torch
        logger.info(f"  PyTorch: {torch.__version__}")
        if torch.backends.mps.is_available():
            logger.info("  MPS backend: ✅ Available")
        elif torch.cuda.is_available():
            logger.info(f"  CUDA: ✅ {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("  ⚠️ No GPU detected, will use CPU (very slow)")
    except ImportError:
        logger.error("❌ PyTorch not installed. Run: pip install -r requirements.txt")
        return False

    try:
        import diffusers
        logger.info(f"  Diffusers: {diffusers.__version__}")
    except ImportError:
        logger.error("❌ Diffusers not installed")
        return False

    try:
        import streamlit
        logger.info(f"  Streamlit: {streamlit.__version__}")
    except ImportError:
        logger.error("❌ Streamlit not installed")
        return False

    # 4. Download model (optional - can be done on first generation)
    logger.info("\n" + "=" * 50)
    logger.info("Model download (stabilityai/stable-video-diffusion-img2vid)")
    logger.info("This is ~10GB and will download on first use.")
    
    response = input("Download model now? [y/N]: ").strip().lower()
    if response == "y":
        logger.info("Downloading model... (this may take several minutes)")
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        del pipe
        logger.info("✅ Model downloaded and cached")
    else:
        logger.info("Skipped. Model will download on first generation.")

    # 5. Train ACSA (optional)
    logger.info("\n" + "=" * 50)
    response = input("Train ACSA classifier? [y/N]: ").strip().lower()
    if response == "y":
        from intelligence.acsa.train_acsa import train
        train()
    else:
        logger.info("Skipped. ACSA will use rule-based fallback.")

    logger.info("\n" + "=" * 50)
    logger.info("✅ Setup complete!")
    logger.info("Run the app: streamlit run frontend/app.py")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
