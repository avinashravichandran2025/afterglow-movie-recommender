from typing import Dict, List, Set, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from afterglow.collaborative_model import CollaborativeRecommender
from afterglow.content_model import ContentRecommender
from afterglow.evaluation import evaluate_model


def split_ratings_train_test(
    ratings_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits ratings dataset into training and testing sets."""
    train, test = train_test_split(ratings_df, test_size=test_size, random_state=random_state)
    return train, test


def prepare_ground_truth(test_df: pd.DataFrame, rating_threshold: float = 3.5) -> Dict[int, Set[int]]:
    """
    Creates ground truth map of user_id -> set of relevant movie_ids
    (movies rated >= rating_threshold).
    """
    relevant = test_df[test_df["rating"] >= rating_threshold]
    ground_truth = {}
    for user_id, group in relevant.groupby("userId"):
        ground_truth[int(user_id)] = set(group["movieId"].astype(int).values)
    return ground_truth


def run_experiments(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    top_k: int = 10,
    n_components: int = 20
) -> pd.DataFrame:
    """
    Runs benchmark experiments comparing Content-Based Filtering
    and Collaborative Filtering models.
    """
    print(f"--- Running Recommendation Experiments (Top-K={top_k}) ---")

    # 1. Train/Test split
    train_df, test_df = split_ratings_train_test(ratings_df, test_size=0.2)
    ground_truth = prepare_ground_truth(test_df)
    test_users = list(ground_truth.keys())

    # 2. Fit Content Model
    print("Fitting Content-Based Model...")
    content_model = ContentRecommender()
    content_model.fit(movies_df, id_col="movieId")

    # 3. Fit Collaborative Model
    print("Fitting Collaborative SVD Model...")
    collab_model = CollaborativeRecommender(n_components=n_components)
    collab_model.fit(train_df)

    # 4. Generate Recommendations on Test Users
    content_recs: Dict[int, List[int]] = {}
    collab_recs: Dict[int, List[int]] = {}

    for user_id in test_users[:100]:  # Evaluate top 100 test users for speed
        # Collaborative recommendations
        c_recs = collab_model.recommend_for_user(user_id=user_id, top_n=top_k)
        collab_recs[user_id] = [int(m) for m in c_recs["movieId"].values] if not c_recs.empty else []

        # Content recommendations (using user's highest rated train movie as seed)
        user_train = train_df[train_df["userId"] == user_id]
        if not user_train.empty:
            seed_movie_id = user_train.sort_values(by="rating", ascending=False).iloc[0]["movieId"]
            if seed_movie_id in content_model.id_to_index:
                cnt_recs = content_model.recommend_similar(movie_id=seed_movie_id, top_n=top_k)
                content_recs[user_id] = [int(m) for m in cnt_recs["movieId"].values] if not cnt_recs.empty else []

    # 5. Evaluate Metrics
    eval_content = evaluate_model(content_recs, ground_truth, k=top_k)
    eval_collab = evaluate_model(collab_recs, ground_truth, k=top_k)

    results_df = pd.DataFrame([
        {"Model": "Content-Based (TF-IDF)", **eval_content},
        {"Model": "Collaborative (SVD)", **eval_collab}
    ])

    return results_df
