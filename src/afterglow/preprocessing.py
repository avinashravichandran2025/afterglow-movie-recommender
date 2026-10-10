
import re
from typing import Any, List, Optional, Tuple

import pandas as pd


TMDB_GENRE_MAP = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}


def clean_genres(genres_str: Any) -> List[str]:
    """Convert pipe-separated genre strings into a list of genres."""
    if not isinstance(genres_str, str):
        return []

    value = genres_str.strip()

    if not value or value == "(no genres listed)":
        return []

    return [genre.strip() for genre in value.split("|") if genre.strip()]


def extract_release_year(
    title: Any,
) -> Tuple[str, Optional[int]]:
    """Extract a four-digit year from a MovieLens-style title."""
    if not isinstance(title, str):
        return "", None

    title = title.strip()
    match = re.search(r"^(.*?)\s*\((\d{4})\)\s*$", title)

    if match:
        return match.group(1).strip(), int(match.group(2))

    return title, None


def _parse_genres(value: Any) -> List[str]:
    """Normalize genre strings, TMDB genre dictionaries, or genre IDs."""
    if isinstance(value, str):
        return clean_genres(value)

    if not isinstance(value, (list, tuple, set)):
        return []

    genres = []

    for item in value:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                genres.append(name.strip())

        elif isinstance(item, int):
            name = TMDB_GENRE_MAP.get(item)
            if name:
                genres.append(name)

        elif isinstance(item, str) and item.strip():
            # Convert numeric strings representing TMDB genre IDs.
            if item.strip().isdigit():
                name = TMDB_GENRE_MAP.get(int(item.strip()))
                if name:
                    genres.append(name)
            else:
                genres.append(item.strip())

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(genres))


def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Clean ratings and retain valid user/movie IDs and rating values."""
    if df.empty:
        return df.copy()

    required = {"userId", "movieId", "rating"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Ratings data is missing required columns: {sorted(missing)}"
        )

    cleaned = df.copy()

    for column in ("userId", "movieId", "rating"):
        cleaned[column] = pd.to_numeric(
            cleaned[column], errors="coerce"
        )

    cleaned = cleaned.dropna(
        subset=["userId", "movieId", "rating"]
    )

    # IDs must be positive whole numbers.
    valid_ids = (
        (cleaned["userId"] > 0)
        & (cleaned["movieId"] > 0)
        & (cleaned["userId"] % 1 == 0)
        & (cleaned["movieId"] % 1 == 0)
    )
    cleaned = cleaned.loc[valid_ids].copy()

    # Discard invalid ratings rather than converting them into valid ratings.
    cleaned = cleaned.loc[
        cleaned["rating"].between(0.5, 5.0)
    ].copy()

    cleaned["userId"] = cleaned["userId"].astype(int)
    cleaned["movieId"] = cleaned["movieId"].astype(int)
    cleaned["rating"] = cleaned["rating"].astype(float)

    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_numeric(
            cleaned["timestamp"], errors="coerce"
        )

    return cleaned.reset_index(drop=True)


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize MovieLens or TMDB movie metadata."""
    if df.empty:
        return df.copy()

    cleaned = df.copy()

    # Ensure a title column exists.
    if "title" not in cleaned.columns:
        if "original_title" in cleaned.columns:
            cleaned["title"] = cleaned["original_title"]
        else:
            raise ValueError(
                "Movie data must contain a 'title' column."
            )

    # Extract MovieLens-style years from titles.
    parsed = cleaned["title"].apply(extract_release_year)
    cleaned["clean_title"] = parsed.apply(lambda item: item[0])
    title_years = parsed.apply(lambda item: item[1])

    # Prefer an explicit TMDB release date when available.
    if "release_date" in cleaned.columns:
        release_dates = pd.to_datetime(
            cleaned["release_date"], errors="coerce"
        )
        date_years = release_dates.dt.year

        cleaned["year"] = title_years.where(
            title_years.notna(), date_years
        )
    else:
        cleaned["year"] = title_years

    cleaned["year"] = cleaned["year"].astype("Int64")

    # Normalize genre data regardless of source format.
    if "genres" in cleaned.columns:
        cleaned["genre_list"] = cleaned["genres"].apply(
            _parse_genres
        )
    elif "genre_ids" in cleaned.columns:
        cleaned["genre_list"] = cleaned["genre_ids"].apply(
            _parse_genres
        )
    else:
        cleaned["genre_list"] = [
            [] for _ in range(len(cleaned))
        ]

    return cleaned
