import pandas as pd
import pytest
from afterglow.mood_recommender import MoodRecommender


@pytest.fixture
def candidate_movies_df():
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4],
        "title": ["Toy Story", "The Conjuring", "The Matrix", "When Harry Met Sally"],
        "genres": ["Animation|Children|Comedy", "Horror|Thriller", "Science Fiction|Action", "Romance|Comedy"]
    })


def test_mood_score_calculation():
    recommender = MoodRecommender()
    
    # Feel good mood should score positive on Animation/Comedy and negative on Horror
    feel_good = recommender.calculate_mood_score("Animation|Comedy", "Uplifting / Feel-Good")
    assert feel_good > 0.5

    horror = recommender.calculate_mood_score("Horror|Thriller", "Uplifting / Feel-Good")
    assert horror < 0.0


def test_watched_history_filtering(candidate_movies_df):
    recommender = MoodRecommender()
    # User has already watched Movie 1 and 3
    recs = recommender.recommend(
        candidate_movies=candidate_movies_df,
        watched_ids=[1, 3],
        top_n=5
    )
    assert 1 not in recs["movieId"].values
    assert 3 not in recs["movieId"].values
    assert set(recs["movieId"].values) == {2, 4}


def test_hybrid_ranking(candidate_movies_df):
    recommender = MoodRecommender()
    base_scores = {1: 0.9, 2: 0.8, 3: 0.85, 4: 0.7}
    
    # Under Uplifting mood, Toy Story (1) and When Harry Met Sally (4) should rank higher
    recs = recommender.recommend(
        candidate_movies=candidate_movies_df,
        mood="Uplifting / Feel-Good",
        base_scores=base_scores,
        top_n=4
    )
    top_movie = recs.iloc[0]["title"]
    assert top_movie in ["Toy Story", "When Harry Met Sally"]
