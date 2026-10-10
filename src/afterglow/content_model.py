
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MovieID = Union[int, str]


class ContentRecommender:
    """
    Content-based movie recommender using TF-IDF and cosine similarity.

    Uses movie genres and overview text. Similarities are computed only
    when needed instead of storing a full movie-by-movie matrix.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.movies_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.similarity_matrix = None  # Retained for interface compatibility
        self.id_to_index: Dict[MovieID, int] = {}
        self.index_to_id: Dict[int, MovieID] = {}
        self.id_col = "movieId"

    @staticmethod
    def _valid_text(value) -> str:
        """Convert a value to safe, readable text."""
        if value is None:
            return ""

        if isinstance(value, (list, tuple, set)):
            return " ".join(
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            )

        if isinstance(value, dict):
            return str(value.get("name", "")).strip()

        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass

        return str(value).strip()

    def _build_feature_text(self, row: pd.Series) -> str:
        """Combine normalized genres and movie overview."""
        genres = row.get("genre_list", None)

        if genres is None:
            genres = row.get("genres", "")

        genre_text = self._valid_text(genres).replace("|", " ")
        overview_text = self._valid_text(row.get("overview", ""))

        # Give genre information additional influence in TF-IDF.
        return f"{genre_text} {genre_text} {overview_text}".strip()

    def fit(
        self,
        movies_df: pd.DataFrame,
        id_col: str = "movieId",
    ) -> "ContentRecommender":
        """Fit TF-IDF features for the supplied movie dataset."""
        if movies_df is None or movies_df.empty:
            raise ValueError(
                "Cannot fit ContentRecommender on an empty DataFrame."
            )

        if id_col not in movies_df.columns:
            raise ValueError(
                f"Movie data must contain the ID column '{id_col}'."
            )

        cleaned = movies_df.copy().reset_index(drop=True)

        if cleaned[id_col].isna().any():
            raise ValueError("Movie IDs must not contain missing values.")

        if cleaned[id_col].duplicated().any():
            raise ValueError("Movie IDs must be unique.")

        self.id_col = id_col
        self.movies_df = cleaned
        self.id_to_index = {
            movie_id: index
            for index, movie_id in enumerate(cleaned[id_col])
        }
        self.index_to_id = {
            index: movie_id
            for movie_id, index in self.id_to_index.items()
        }

        feature_text = cleaned.apply(
            self._build_feature_text,
            axis=1,
        ).fillna("")

        # Empty descriptions need a harmless fallback token.
        feature_text = feature_text.apply(
            lambda text: text if text.strip() else "unknown"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(feature_text)
        self.similarity_matrix = None

        return self

    def recommend_similar(
        self,
        movie_id: MovieID,
        top_n: int = 10,
        exclude_ids: Optional[List[MovieID]] = None,
    ) -> pd.DataFrame:
        """Return the most similar movies, excluding specified IDs."""
        if self.movies_df is None or self.tfidf_matrix is None:
            raise RuntimeError(
                "Model is not fitted yet. Call fit() first."
            )

        if top_n <= 0:
            return self.movies_df.iloc[0:0].copy()

        if movie_id not in self.id_to_index:
            raise KeyError(
                f"Movie ID '{movie_id}' not found in fitted dataset."
            )

        target_index = self.id_to_index[movie_id]

        # Calculate similarities against this movie only.
        scores = cosine_similarity(
            self.tfidf_matrix[target_index],
            self.tfidf_matrix,
        ).ravel()

        excluded = set(exclude_ids or [])
        excluded.add(movie_id)

        # Sort highest similarity first.
        ordered_indices = np.argsort(-scores, kind="stable")

        recommendations = []

        for index in ordered_indices:
            candidate_id = self.index_to_id[int(index)]

            if candidate_id in excluded:
                continue

            row = self.movies_df.iloc[int(index)].to_dict()
            row["similarity_score"] = round(
                float(scores[index]), 4
            )
            recommendations.append(row)

            if len(recommendations) >= top_n:
                break

        return pd.DataFrame(recommendations)
