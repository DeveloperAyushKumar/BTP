"""Database - SQLite setup and connection management."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ecovideo.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def initialize_database():
    """Create all tables if they don't exist."""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                preset TEXT NOT NULL,
                seed INTEGER,
                generation_time REAL,
                peak_vram REAL,
                ssim REAL,
                psnr REAL,
                energy_kwh REAL,
                carbon_gco2 REAL,
                output_video_path TEXT,
                user_rating INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_generations_timestamp 
                ON generations(timestamp);

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                preset TEXT NOT NULL,
                generation_time REAL,
                peak_vram REAL,
                ssim REAL,
                psnr REAL,
                energy_wh REAL,
                is_optimal INTEGER DEFAULT 0
            );
        """)
        conn.commit()
        logger.info(f"Database initialized at {DB_PATH}")
    finally:
        conn.close()
