
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


MovieID = Union[int, str]


class CollaborativeRecommender:
    """
    Collaborative filtering baseline using TruncatedSVD.

    Missing ratings are represented by zero in the matrix. This is a
    simplified baseline; zero-filled SVD does not explicitly model
    missing ratings as unknown values.
    """

    def __init__(self, n_components: int = 20):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")

        self.n_components = n_components
        self.svd: Optional[TruncatedSVD] = None

        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.reconstructed_matrix: Optional[np.ndarray] = None

        self.user_to_idx: Dict[int, int] = {}
        self.movie_to_idx: Dict[MovieID, int] = {}
        self.idx_to_movie: Dict[int, MovieID] = {}

        self.mean_rating = 3.5

    def fit(self, ratings_df: pd.DataFrame) -> "CollaborativeRecommender":
        """Fit the SVD model on user–movie ratings."""
        if ratings_df is None or ratings_df.empty:
            raise ValueError("Cannot fit on an empty ratings DataFrame.")

        required = {"userId", "movieId", "rating"}
        missing = required - set(ratings_df.columns)

        if missing:
            raise ValueError(
                f"Ratings data is missing columns: {sorted(missing)}"
            )

        ratings = ratings_df[["userId", "movieId", "rating"]].copy()

        ratings["userId"] = pd.to_numeric(
            ratings["userId"], errors="coerce"
        )
        ratings["rating"] = pd.to_numeric(
            ratings["rating"], errors="coerce"
        )

        ratings = ratings.dropna(
            subset=["userId", "movieId", "rating"]
        )

        ratings = ratings[
            np.isfinite(ratings["userId"])
            & np.isfinite(ratings["rating"])
            & (ratings["userId"] > 0)
            & (ratings["userId"] % 1 == 0)
            & ratings["rating"].between(0.5, 5.0)
        ].copy()

        if ratings.empty:
            raise ValueError("No valid ratings remain after cleaning.")

        # Keep IDs consistent with the original MovieLens format.
        ratings["userId"] = ratings["userId"].astype(int)

        # Resolve duplicate user–movie ratings by averaging them.
        ratings = ratings.groupby(
            ["userId", "movieId"], as_index=False
        )["rating"].mean()

        self.mean_rating = float(ratings["rating"].mean())

        self.user_item_matrix = ratings.pivot(
            index="userId",
            columns="movieId",
            values="rating",
        ).fillna(0.0)

        self.user_to_idx = {
            user_id: idx
            for idx, user_id in enumerate(self.user_item_matrix.index)
        }

        self.movie_to_idx = {
            movie_id: idx
            for idx, movie_id in enumerate(self.user_item_matrix.columns)
        }

        self.idx_to_movie = {
            idx: movie_id
            for movie_id, idx in self.movie_to_idx.items()
        }

        matrix = self.user_item_matrix.to_numpy(dtype=float)

        # SVD requires a sensible component count for the matrix size.
        min_dimension = min(matrix.shape)

        if min_dimension < 2:
            self.svd = None
            self.reconstructed_matrix = matrix.copy()
            return self

        actual_components = min(
            self.n_components,
            min_dimension - 1,
        )

        self.svd = TruncatedSVD(
            n_components=actual_components,
            random_state=42,
        )

        reduced_matrix = self.svd.fit_transform(matrix)
        self.reconstructed_matrix = (
            reduced_matrix @ self.svd.components_
        )

        return self

    def predict_rating(
        self,
        user_id: int,
        movie_id: MovieID,
    ) -> float:
        """Predict a rating, falling back to the global mean for unknown IDs."""
        if self.reconstructed_matrix is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if (
            user_id not in self.user_to_idx
            or movie_id not in self.movie_to_idx
        ):
            return round(self.mean_rating, 2)

        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]

        predicted = self.reconstructed_matrix[user_idx, movie_idx]

        return round(float(np.clip(predicted, 0.5, 5.0)), 2)

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_ids: Optional[List[MovieID]] = None,
    ) -> pd.DataFrame:
        """Recommend unrated movies using reconstructed rating scores."""
        if self.reconstructed_matrix is None or self.user_item_matrix is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        columns = ["movieId", "predicted_rating"]

        if top_n <= 0:
            return pd.DataFrame(columns=columns)

        excluded = set(exclude_ids or [])

        if user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            scores = self.reconstructed_matrix[user_idx]

            rated_movies = self.user_item_matrix.columns[
                self.user_item_matrix.iloc[user_idx].to_numpy() > 0
            ]
            excluded.update(rated_movies)
        else:
            # Unseen-user fallback: average predicted score per movie.
            scores = np.mean(self.reconstructed_matrix, axis=0)

        results = []

        for movie_idx, score in enumerate(scores):
            movie_id = self.idx_to_movie[movie_idx]

            if movie_id in excluded:
                continue

            results.append({
                "movieId": movie_id,
                "predicted_rating": round(
                    float(np.clip(score, 0.5, 5.0)), 2
                ),
            })

        results.sort(
            key=lambda item: item["predicted_rating"],
            reverse=True,
        )

        return pd.DataFrame(results[:top_n], columns=columns)
