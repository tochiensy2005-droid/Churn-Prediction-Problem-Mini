"""src/config.py

Reads configuration settings from environment and the .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Parse .env manually to avoid external dependencies
env_dict = {}
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env_dict[k.strip()] = v.strip().strip('"').strip("'")


def get_env(key: str, default: str) -> str:
    return os.getenv(key, env_dict.get(key, default))


MINIO_ENDPOINT = get_env("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = get_env("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = get_env("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = get_env("MINIO_BUCKET", "churn-prediction")

DATA_DIR = Path(get_env("DATA_DIR", str(ROOT)))
OUTPUT_DIR = Path(get_env("OUTPUT_DIR", str(ROOT / "output")))
ARTIFACTS_DIR = Path(get_env("ARTIFACTS_DIR", str(ROOT / "artifacts")))

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_WINDOW_MONTHS = int(get_env("TRAINING_WINDOW_MONTHS", "12"))
DEFAULT_THRESHOLD = float(get_env("DEFAULT_THRESHOLD", "0.94"))
MODEL_VERSION = get_env("MODEL_VERSION", "1.0.0")
FEATURE_VERSION = get_env("FEATURE_VERSION", "1.0.0")
