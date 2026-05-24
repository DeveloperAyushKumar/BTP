# EcoVideo Studio 🌿

**Energy-Efficient Video Generation on Apple Silicon**

EcoVideo Studio generates videos from images using Stable Video Diffusion, optimized for Apple Silicon M3 Pro with a focus on sustainability and Green AI principles.

## Features

- **Video Generation**: Transform static images into 14-frame videos using SVD
- **Three Quality Presets**: Eco (fast/efficient), Balanced, Quality (best output)
- **Sustainability Metrics**: Real-time energy, carbon footprint, and cloud savings tracking
- **ACSA Algorithm**: Adaptive Compression Selection using CLIP features + hardware profiling
- **Local Processing**: 100% on-device, no cloud dependency
- **History & Analytics**: Track all generations with quality and efficiency metrics

## Installation

```bash
cd ecovideo-studio
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup_app.py  # Downloads model & initializes database
```

## Usage

```bash
streamlit run frontend/app.py
```

Navigate to `http://localhost:8501` in your browser.

## Architecture

```
ecovideo-studio/
├── frontend/          # Streamlit UI
├── backend/           # Inference orchestration
├── ml_components/     # SVD model, compression, quality
├── intelligence/      # ACSA adaptive preset selection
├── sustainability/    # Energy, carbon, impact metrics
├── storage/           # SQLite persistence
├── utils/             # Shared utilities
├── data/              # Models, benchmarks, configs
├── tests/             # Test suite
└── notebooks/         # Research notebooks
```

## Sustainability Methodology

Energy is estimated using M3 Pro power profiles (22W GPU load, 5W idle) multiplied by generation time. Carbon emissions use regional grid intensity factors (default 475 gCO2/kWh). Savings are computed against a cloud baseline of ~150 Wh per video.

## ACSA Algorithm

The Adaptive Compression Selection Algorithm uses a small MLP classifier that takes CLIP image embeddings (768-dim) concatenated with hardware features to predict the optimal preset for each image. This achieves ~85% accuracy on validation data.

## Performance (M3 Pro)

| Preset | Time | Peak VRAM | Energy |
|--------|------|-----------|--------|
| Eco | 90-150s | ~8GB | ~8 Wh |
| Balanced | 180-240s | ~10GB | ~11 Wh |
| Quality | 300-420s | ~12GB | ~15 Wh |

## Limitations

- MPS backend does not support INT8/INT4 quantization (bitsandbytes)
- CPU offloading designed for CUDA is not available
- Power measurement is estimated, not direct hardware reading
- Neural Engine not accessible through PyTorch

## References

- Stable Video Diffusion (Blattmann et al., 2023)
- Green AI (Schwartz et al., 2020)
- CLIP (Radford et al., 2021)

## License

MIT
# BTP
