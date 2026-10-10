import pytest
from afterglow import config

def test_config_paths():
    assert config.BASE_DIR.exists()
    assert config.DATA_DIR.exists()
    assert config.RAW_DATA_DIR.exists()
    assert config.PROCESSED_DATA_DIR.exists()

def test_tmdb_auth_helpers():
    headers = config.get_tmdb_headers()
    assert isinstance(headers, dict)
    assert "accept" in headers

    params = config.get_tmdb_params({"page": 1})
    assert params["page"] == 1
