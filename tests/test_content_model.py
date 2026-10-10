import pandas as pd
import pytest
from afterglow.content_model import ContentRecommender


@pytest.fixture
def sample_movies_df():
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4],
        "title": ["Toy Story", "Jumanji", "Grumpier Old Men", "Toy Story 2"],
        "genres": ["Animation|Children|Comedy", "Adventure|Children|Fantasy", "Comedy|Romance", "Animation|Children|Comedy"],
        "overview": [
            "A story about toys that come to life.",
            "A magical board game that unleashes jungle hazards.",
            "Two old friends argue and fall in love.",
            "Woody is stolen by a toy collector."
        ]
    })


def test_fit_and_recommend(sample_movies_df):
    model = ContentRecommender()
    model.fit(sample_movies_df, id_col="movieId")

    # Toy Story (1) should be most similar to Toy Story 2 (4)
    recs = model.recommend_similar(movie_id=1, top_n=2)
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) == 2
    assert recs.iloc[0]["movieId"] == 4  # Toy Story 2
    assert recs.iloc[0]["similarity_score"] > 0.20


def test_exclude_ids(sample_movies_df):
    model = ContentRecommender()
    model.fit(sample_movies_df, id_col="movieId")

    recs = model.recommend_similar(movie_id=1, top_n=2, exclude_ids=[4])
    assert 4 not in recs["movieId"].values


def test_unfitted_model_error():
    model = ContentRecommender()
    with pytest.raises(RuntimeError):
        model.recommend_similar(movie_id=1)


def test_missing_movie_id_error(sample_movies_df):
    model = ContentRecommender()
    model.fit(sample_movies_df, id_col="movieId")
    with pytest.raises(KeyError):
        model.recommend_similar(movie_id=999)
