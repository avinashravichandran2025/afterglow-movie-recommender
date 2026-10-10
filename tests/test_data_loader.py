import pandas as pd
import pytest
from afterglow import data_loader


def test_tmdb_connection_verification():
    """Verify live authentication against TMDB API."""
    assert data_loader.verify_tmdb_connection() is True


def test_fetch_popular_movies():
    """Verify fetching live popular movies metadata from TMDB."""
    movies = data_loader.fetch_popular_movies(pages=1)
    assert isinstance(movies, list)
    assert len(movies) > 0
    first_movie = movies[0]
    assert "id" in first_movie
    assert "title" in first_movie
    assert "overview" in first_movie


def test_load_movielens_ratings(tmp_path):
    """Test loading MovieLens ratings from a sample CSV."""
    sample_csv = tmp_path / "sample_ratings.csv"
    sample_csv.write_text("userId,movieId,rating,timestamp\n1,101,4.5,12345678\n2,102,3.0,12345679\n")

    df = data_loader.load_movielens_ratings(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["userId", "movieId", "rating", "timestamp"]


def test_load_movielens_movies(tmp_path):
    """Test loading MovieLens movies from a sample CSV."""
    sample_csv = tmp_path / "sample_movies.csv"
    sample_csv.write_text("movieId,title,genres\n101,Toy Story (1995),Animation|Children|Comedy\n")

    df = data_loader.load_movielens_movies(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Toy Story (1995)"
