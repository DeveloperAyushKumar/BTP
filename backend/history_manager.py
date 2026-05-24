"""History Manager - SQLite operations for generation history."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from storage.database import get_connection

logger = logging.getLogger(__name__)


@dataclass
class GenerationRecord:
    """A stored generation record."""
    id: str
    timestamp: str
    image_path: str
    preset: str
    seed: int
    generation_time: float
    peak_vram: float
    ssim: Optional[float]
    psnr: Optional[float]
    energy_kwh: float
    carbon_gco2: float
    output_video_path: str
    user_rating: Optional[int] = None


class HistoryManager:
    """Manages generation history in SQLite."""

    def add_generation(
        self,
        generation_id: str,
        image_path: str,
        preset: str,
        seed: int,
        generation_time: float,
        peak_vram: float,
        ssim: Optional[float],
        psnr: Optional[float],
        energy_kwh: float,
        carbon_gco2: float,
        output_video_path: str,
    ):
        """Store a completed generation."""
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO generations 
                (id, timestamp, image_path, preset, seed, generation_time,
                 peak_vram, ssim, psnr, energy_kwh, carbon_gco2, output_video_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    generation_id,
                    datetime.now().isoformat(),
                    image_path,
                    preset,
                    seed,
                    generation_time,
                    peak_vram,
                    ssim,
                    psnr,
                    energy_kwh,
                    carbon_gco2,
                    output_video_path,
                ),
            )
            conn.commit()
            logger.info(f"Stored generation: {generation_id}")
        finally:
            conn.close()

    def get_recent_generations(self, limit: int = 20) -> list[GenerationRecord]:
        """Get most recent generations."""
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM generations ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [GenerationRecord(*row) for row in rows]
        finally:
            conn.close()

    def get_total_savings(self) -> dict:
        """Get cumulative sustainability savings."""
        conn = get_connection()
        try:
            row = conn.execute(
                """SELECT 
                    COUNT(*) as total_generations,
                    COALESCE(SUM(energy_kwh), 0) as total_energy,
                    COALESCE(SUM(carbon_gco2), 0) as total_carbon
                FROM generations"""
            ).fetchone()
            return {
                "total_generations": row[0],
                "total_energy_kwh": row[1],
                "total_carbon_gco2": row[2],
            }
        finally:
            conn.close()

    def delete_generation(self, generation_id: str):
        """Delete a generation record."""
        conn = get_connection()
        try:
            conn.execute("DELETE FROM generations WHERE id = ?", (generation_id,))
            conn.commit()
        finally:
            conn.close()

    def update_rating(self, generation_id: str, rating: int):
        """Update user rating for a generation."""
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE generations SET user_rating = ? WHERE id = ?",
                (rating, generation_id),
            )
            conn.commit()
        finally:
            conn.close()
