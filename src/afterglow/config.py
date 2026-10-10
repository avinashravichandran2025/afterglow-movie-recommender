
import os
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Project root: config.py is inside src/afterglow/
BASE_DIR = Path(__file__).resolve().parents[2]

# Load local environment variables for development
load_dotenv(BASE_DIR / ".env")

# Base directory paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure data directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_secret_or_env(name: str) -> str:
    """Read a credential from Streamlit Secrets or environment variables."""
    try:
        value = st.secrets.get(name, "")
    except (FileNotFoundError, RuntimeError):
        # Streamlit Secrets may not exist during local development.
        value = ""

    if value:
        return str(value).strip()

    return os.getenv(name, "").strip()


# TMDB API configuration
TMDB_API_KEY = get_secret_or_env("TMDB_API_KEY")
TMDB_READ_TOKEN = get_secret_or_env("TMDB_READ_TOKEN")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


def get_tmdb_headers() -> dict:
    """Return headers for TMDB API requests."""
    headers = {
        "accept": "application/json",
        "User-Agent": "Afterglow-Movie-Recommender/1.0",
    }

    if TMDB_READ_TOKEN:
        headers["Authorization"] = f"Bearer {TMDB_READ_TOKEN}"

    return headers


def get_tmdb_params(
    extra_params: Optional[dict] = None,
) -> dict:
    """Return query parameters, adding the API key when needed."""
    params = dict(extra_params or {})

    # Prefer the Read Access Token when configured.
    if not TMDB_READ_TOKEN and TMDB_API_KEY:
        params["api_key"] = TMDB_API_KEY

    return params


def validate_config() -> bool:
    """Check whether at least one TMDB credential is configured."""
    return bool(TMDB_READ_TOKEN or TMDB_API_KEY)
