"""ACSA Training Pipeline - Train the adaptive preset classifier."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from intelligence.acsa.classifier import ACSAClassifier, PRESET_LABELS
from backend.config_manager import TRAINED_MODELS_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARK_PATH = DATA_DIR / "benchmarks" / "phase_3_to_6_data.csv"
MODEL_SAVE_PATH = TRAINED_MODELS_DIR / "acsa_classifier.pth"


def generate_synthetic_features(num_samples_per_row: int = 10, feature_dim: int = 768) -> tuple:
    """
    Generate training data from benchmark CSV.
    Since we don't have actual CLIP features for images, we synthesize them.
    In production, you'd extract real CLIP features from test images.
    """
    df = pd.read_csv(BENCHMARK_PATH)

    features_list = []
    labels_list = []

    preset_to_idx = {name: idx for idx, name in enumerate(PRESET_LABELS)}

    for _, row in df.iterrows():
        if row["is_optimal"] != 1:
            continue

        # Synthesize CLIP-like features based on image properties
        # Use SSIM and complexity as seeds for reproducible random features
        seed = hash(row["image_path"]) % (2**31)
        rng = np.random.RandomState(seed)

        for _ in range(num_samples_per_row):
            # Random CLIP features (768-dim normalized)
            clip_feat = rng.randn(feature_dim).astype(np.float32)
            clip_feat = clip_feat / np.linalg.norm(clip_feat)

            # Hardware features (4-dim)
            hw_feat = np.array([
                rng.uniform(8, 20),    # available_vram_gb
                rng.uniform(0.5, 0.9), # compute_score
                rng.uniform(0.1, 0.6), # cpu_usage normalized
                rng.uniform(0.3, 0.6), # ram normalized
            ], dtype=np.float32)

            full_feature = np.concatenate([clip_feat, hw_feat])
            features_list.append(full_feature)
            labels_list.append(preset_to_idx[row["preset_used"]])

    features = np.array(features_list)
    labels = np.array(labels_list)
    return features, labels


def train():
    """Train the ACSA classifier."""
    logger.info("Generating training data...")
    features, labels = generate_synthetic_features(num_samples_per_row=50)
    logger.info(f"Training data: {features.shape[0]} samples, {features.shape[1]} features")

    # Normalize features
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0) + 1e-8
    features_norm = (features - feature_mean) / feature_std

    # Train/val split (80/20)
    n = len(features_norm)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train = torch.tensor(features_norm[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels[train_idx], dtype=torch.long)
    X_val = torch.tensor(features_norm[val_idx], dtype=torch.float32)
    y_val = torch.tensor(labels[val_idx], dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Initialize model
    model = ACSAClassifier(input_dim=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logger.info("Training for 100 epochs...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = (val_preds == y_val).float().mean().item()
            logger.info(f"Epoch {epoch+1}/100 | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.3f}")

    # Final validation accuracy
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_preds = torch.argmax(val_logits, dim=1)
        final_acc = (val_preds == y_val).float().mean().item()

    logger.info(f"Final validation accuracy: {final_acc:.3f}")

    # Save model with normalization stats
    model._feature_mean = torch.tensor(feature_mean, dtype=torch.float32)
    model._feature_std = torch.tensor(feature_std, dtype=torch.float32)
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
