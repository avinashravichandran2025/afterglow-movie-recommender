import pandas as pd
import pytest
from afterglow import preprocessing


def test_clean_genres():
    assert preprocessing.clean_genres("Animation|Children|Comedy") == ["Animation", "Children", "Comedy"]
    assert preprocessing.clean_genres("(no genres listed)") == []
    assert preprocessing.clean_genres("") == []
    assert preprocessing.clean_genres(None) == []


def test_extract_release_year():
    title, year = preprocessing.extract_release_year("Toy Story (1995)")
    assert title == "Toy Story"
    assert year == 1995

    title_no_year, year_none = preprocessing.extract_release_year("Matrix, The")
    assert title_no_year == "Matrix, The"
    assert year_none is None


def test_clean_ratings():
    raw_data = pd.DataFrame({
        "userId": [1, 2, None, 4],
        "movieId": [101, 102, 103, 104],
        "rating": [4.5, 6.0, 3.0, -1.0],
        "timestamp": [1000, 2000, 3000, 4000]
    })
    cleaned = preprocessing.clean_ratings(raw_data)
    assert len(cleaned) == 3
    assert cleaned.iloc[1]["rating"] == 5.0  # Clipped to 5.0
    assert cleaned.iloc[2]["rating"] == 0.5  # Clipped to 0.5


def test_clean_movies():
    raw_movies = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Toy Story (1995)", "Jumanji (1995)"],
        "genres": ["Animation|Children|Comedy", "Adventure|Children|Fantasy"]
    })
    cleaned = preprocessing.clean_movies(raw_movies)
    assert "clean_title" in cleaned.columns
    assert "year" in cleaned.columns
    assert "genre_list" in cleaned.columns

    assert cleaned.iloc[0]["clean_title"] == "Toy Story"
    assert cleaned.iloc[0]["year"] == 1995
    assert cleaned.iloc[0]["genre_list"] == ["Animation", "Children", "Comedy"]
