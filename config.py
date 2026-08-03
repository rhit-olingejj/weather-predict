from __future__ import annotations

import os
from pathlib import Path

# Where the observation CSV written by fetch_weather.py lives
DEFAULT_DATA = "weather_daily.csv"

# Where xg.py writes the trained bundle and predict_weather.py looks for it
DEFAULT_MODEL_DIR = "models"
MODEL_FILENAME = "xgb_weather.joblib"

# A day counts as "wet" above the WMO trace threshold; below that, gauges are mostly reporting dew and condensation rather than real precipitation. Both sides need this: training turns it into the label, serving reports it
PRECIP_THRESHOLD_MM = 0.1

# Database credentials live in a gitignored .env next to this file (copy .env.example). Loaded here because both training and serving import config, and python-dotenv is optional -- without it, real environment variables still work.
DOTENV_PATH = Path(__file__).with_name(".env")


def load_env(path: Path = DOTENV_PATH) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # override=False: a variable already set in the real environment wins over the file, which is what CI and container deployments expect.
    load_dotenv(path, override=False)


load_env()

# Where observations are read from: "csv" reads DEFAULT_DATA, "db" reads the SQL database configured above. Overridden per run by --source.
DATA_SOURCES = ("csv", "db")
DEFAULT_DATA_SOURCE = (os.getenv("DATA_SOURCE") or "csv").strip().lower()
