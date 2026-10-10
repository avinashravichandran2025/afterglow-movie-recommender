from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


MovieID = Union[int, str]


MOOD_PROFILES: Dict[str, Dict[str, float]] = {
    "Uplifting / Feel-Good": {
        "Comedy": 1.0,
        "Animation": 0.8,
        "Family": 0.8,
        "Romance": 0.6,
        "Music": 0.5,
        "Horror": -0.8,
        "Thriller": -0.5,
        "Drama": 0.0,
    },
    "Dark / Gritty / Intense": {
        "Thriller": 1.0,
        "Crime": 0.9,
        "Horror": 0.8,
        "Mystery": 0.7,
        "Action": 0.6,
        "Comedy": -0.7,
        "Animation": -0.6,
        "Family": -0.9,
    },
    "Mind-Bending / Sci-Fi": {
        "Science Fiction": 1.0,
        "Mystery": 0.9,
        "Thriller": 0.7,
        "Fantasy": 0.6,
        "Comedy": -0.2,
        "Romance": -0.2,
    },
    "Relaxed / Chill": {
        "Animation": 0.9,
        "Documentary": 0.8,
        "Comedy": 0.7,
        "Family": 0.7,
        "Horror": -1.0,
        "Action": -0.5,
    },
    "Romantic / Heartwarming": {
        "Romance": 1.0,
        "Drama": 0.6,
        "Comedy": 0.6,
        "Horror": -0.9,
        "Action": -0.4,
    },
}


class MoodRecommender:
    """Rank movies using content similarity and mood alignment."""

    def __init__(
        self,
        mood_profiles: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.mood_profiles = mood_profiles or MOOD_PROFILES

    @staticmethod
    def _normalize_id(value: MovieID) -> str:
        """Make integer and string movie IDs comparable."""
        if isinstance(value, (int, np.integer)):
            return str(int(value))

        if isinstance(value, (float, np.floating)) and value.is_integer():
            return str(int(value))

        return str(value).strip()

    @staticmethod
    def _parse_genres(genres) -> List[str]:
        """Read genre names from lists, strings, or TMDB dictionaries."""
        if isinstance(genres, str):
            return [g.strip() for g in genres.split("|") if g.strip()]

        if not isinstance(genres, (list, tuple, set)):
            return []

        result = []
        for genre in genres:
            if isinstance(genre, dict):
                name = genre.get("name")
                if isinstance(name, str) and name.strip():
                    result.append(name.strip())
            elif isinstance(genre, str) and genre.strip():
                result.append(genre.strip())

        return result

    def calculate_mood_score(self, genres, mood: str) -> float:
        """Calculate genre alignment for a selected mood."""
        if mood not in self.mood_profiles:
            return 0.0

        genre_list = self._parse_genres(genres)
        if not genre_list:
            return 0.0

        weights = self.mood_profiles[mood]
        return float(np.mean([weights.get(g, 0.0) for g in genre_list]))

    def recommend(
        self,
        candidate_movies: pd.DataFrame,
        mood: Optional[str] = None,
        watched_ids: Optional[List[MovieID]] = None,
        base_scores: Optional[Dict[MovieID, float]] = None,
        id_col: str = "movieId",
        top_n: int = 10,
        mood_weight: float = 0.4,
    ) -> pd.DataFrame:
        """Return the highest-ranked eligible movies."""
        if candidate_movies is None or candidate_movies.empty:
            return pd.DataFrame()

        if id_col not in candidate_movies.columns:
            raise ValueError(f"Candidate movies must contain '{id_col}'.")

        if top_n <= 0:
            return candidate_movies.iloc[0:0].copy()

        if not 0.0 <= mood_weight <= 1.0:
            raise ValueError("mood_weight must be between 0 and 1.")

        df = candidate_movies.copy()

        watched = {
            self._normalize_id(movie_id)
            for movie_id in (watched_ids or [])
        }
        df = df[
            ~df[id_col].map(self._normalize_id).isin(watched)
        ].copy()

        if df.empty:
            return df

        # Align content scores with the movie IDs.
        score_map = {
            self._normalize_id(movie_id): float(score)
            for movie_id, score in (base_scores or {}).items()
            if score is not None and np.isfinite(score)
        }

        df["base_score"] = df[id_col].map(
            lambda movie_id: score_map.get(
                self._normalize_id(movie_id), np.nan
            )
        )
        df["base_score"] = pd.to_numeric(
            df["base_score"], errors="coerce"
        )

        available = df["base_score"].dropna()
        fallback = float(available.median()) if not available.empty else 0.5
        df["base_score"] = df["base_score"].fillna(fallback)

        # Convert content scores to [0, 1].
        base_min = df["base_score"].min()
        base_max = df["base_score"].max()

        if base_max > base_min:
            norm_base = (
                (df["base_score"] - base_min) / (base_max - base_min)
            )
        else:
            norm_base = pd.Series(0.5, index=df.index)

        # No selected mood means pure content ranking.
        if mood not in self.mood_profiles:
            df["mood_score"] = 0.0
            df["final_score"] = norm_base
        else:
            genre_column = (
                "genre_list" if "genre_list" in df.columns else "genres"
            )

            if genre_column in df.columns:
                df["mood_score"] = df[genre_column].apply(
                    lambda genres: self.calculate_mood_score(genres, mood)
                )
            else:
                df["mood_score"] = 0.0

            norm_mood = (df["mood_score"].clip(-1, 1) + 1) / 2

            df["final_score"] = (
                (1 - mood_weight) * norm_base
                + mood_weight * norm_mood
            )

        df["final_score"] = df["final_score"].round(4)

        return (
            df.sort_values(
                "final_score",
                ascending=False,
                kind="stable",
            )
            .head(top_n)
            .reset_index(drop=True)
        )
