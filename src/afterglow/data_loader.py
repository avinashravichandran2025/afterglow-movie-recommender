import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from afterglow import config


def fetch_tmdb_api(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fetch JSON data from TMDB with retries for temporary failures."""
    clean_endpoint = endpoint.lstrip("/")
    url = f"{config.TMDB_BASE_URL.rstrip('/')}/{clean_endpoint}"

    headers = config.get_tmdb_headers()
    query_params = config.get_tmdb_params(params)

    last_error: Optional[Exception] = None

    # Retry transient connection failures and temporary server errors.
    with requests.Session() as session:
        for attempt in range(5):
            try:
                response = session.get(
                    url,
                    headers=headers,
                    params=query_params,
                    timeout=(10, 25),
                )

                # Retry rate limits and temporary TMDB server errors.
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < 4:
                        retry_after = response.headers.get("Retry-After")

                        try:
                            delay = (
                                float(retry_after)
                                if retry_after
                                else 2 ** attempt
                            )
                        except (ValueError, TypeError):
                            delay = 2 ** attempt

                        time.sleep(min(delay, 15))
                        continue

                response.raise_for_status()

                data = response.json()

                if not isinstance(data, dict):
                    raise ValueError(
                        "TMDB returned an unexpected JSON format."
                    )

                return data

            except (requests.ConnectionError, requests.Timeout) as err:
                last_error = err

                if attempt < 4:
                    time.sleep(min(2 ** attempt, 10))

            except requests.HTTPError as err:
                status_code = (
                    err.response.status_code
                    if err.response is not None
                    else "unknown"
                )

                raise RuntimeError(
                    f"TMDB returned HTTP {status_code} "
                    f"for endpoint '{clean_endpoint}'. "
                    "Check your API credentials and request parameters."
                ) from err

            except ValueError as err:
                raise RuntimeError(
                    f"TMDB returned invalid JSON for endpoint "
                    f"'{clean_endpoint}': {err}"
                ) from err

    raise RuntimeError(
        f"Could not connect to TMDB endpoint '{clean_endpoint}' "
        f"after 5 attempts. Last connection error: {last_error}"
    )


def verify_tmdb_connection() -> bool:
    """Verify TMDB configuration and test an authenticated API request."""
    if not config.validate_config():
        print("[ERROR] TMDB API key or Read Access Token is missing.")
        return False

    try:
        data = fetch_tmdb_api("movie/popular", params={"page": 1})

        if isinstance(data.get("results"), list):
            print("[SUCCESS] TMDB API connection verified.")
            return True

        print("[ERROR] TMDB response did not contain movie results.")
        return False

    except Exception as err:
        print(f"[ERROR] TMDB connection check failed: {err}")
        return False


def fetch_popular_movies(pages: int = 1) -> List[Dict[str, Any]]:
    """Fetch popular movie metadata from one or more TMDB pages."""
    if pages < 1:
        raise ValueError("pages must be at least 1")

    movies: List[Dict[str, Any]] = []

    for page in range(1, pages + 1):
        data = fetch_tmdb_api(
            "movie/popular",
            params={"page": page},
        )
        movies.extend(data.get("results", []))

    return movies


def fetch_movie_details(movie_id: int) -> Dict[str, Any]:
    """Fetch details for a TMDB movie ID."""
    if movie_id < 1:
        raise ValueError("movie_id must be a positive integer")

    return fetch_tmdb_api(f"movie/{movie_id}")


def load_movielens_ratings(
    filepath: Optional[Path] = None,
) -> pd.DataFrame:
    """Load MovieLens ratings CSV with required-column validation."""
    path = filepath or (config.RAW_DATA_DIR / "ratings.csv")

    if not path.exists():
        raise FileNotFoundError(
            f"MovieLens ratings file not found at: {path}"
        )

    df = pd.read_csv(path)
    expected_cols = {"userId", "movieId", "rating", "timestamp"}

    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Ratings CSV missing required columns: "
            f"{sorted(expected_cols - set(df.columns))}"
        )

    return df


def load_movielens_movies(
    filepath: Optional[Path] = None,
) -> pd.DataFrame:
    """Load MovieLens movies CSV with required-column validation."""
    path = filepath or (config.RAW_DATA_DIR / "movies.csv")

    if not path.exists():
        raise FileNotFoundError(
            f"MovieLens movies file not found at: {path}"
        )

    df = pd.read_csv(path)
    expected_cols = {"movieId", "title", "genres"}

    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Movies CSV missing required columns: "
            f"{sorted(expected_cols - set(df.columns))}"
        )

    return df
