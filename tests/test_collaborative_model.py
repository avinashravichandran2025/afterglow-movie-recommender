import pandas as pd
import pytest
from afterglow.collaborative_model import CollaborativeRecommender


@pytest.fixture
def sample_ratings_df():
    return pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3, 3, 3],
        "movieId": [101, 102, 103, 101, 104, 102, 103, 104],
        "rating": [5.0, 4.0, 1.0, 4.5, 5.0, 1.0, 1.5, 4.5]
    })


def test_fit_and_predict(sample_ratings_df):
    model = CollaborativeRecommender(n_components=2)
    model.fit(sample_ratings_df)

    pred = model.predict_rating(user_id=1, movie_id=101)
    assert 0.5 <= pred <= 5.0


def test_recommend_for_user(sample_ratings_df):
    model = CollaborativeRecommender(n_components=2)
    model.fit(sample_ratings_df)

    # User 1 rated 101, 102, 103 -> should recommend 104
    recs = model.recommend_for_user(user_id=1, top_n=2)
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) == 1
    assert recs.iloc[0]["movieId"] == 104


def test_cold_start_user(sample_ratings_df):
    model = CollaborativeRecommender(n_components=2)
    model.fit(sample_ratings_df)

    # User 999 is unseen -> fallback
    pred = model.predict_rating(user_id=999, movie_id=101)
    assert pred == round(sample_ratings_df["rating"].mean(), 2)

    recs = model.recommend_for_user(user_id=999, top_n=2)
    assert len(recs) == 2


def test_unfitted_error():
    model = CollaborativeRecommender()
    with pytest.raises(RuntimeError):
        model.predict_rating(user_id=1, movie_id=101)
