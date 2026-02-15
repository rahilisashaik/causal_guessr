"""FRED (Federal Reserve Economic Data) API client."""

from api.fred.client import (
    get_observations,
    get_observations_cached,
    get_release_series,
    get_releases_cached,
    get_series,
    search_series,
)

__all__ = [
    "get_observations",
    "get_observations_cached",
    "get_release_series",
    "get_releases_cached",
    "get_series",
    "search_series",
]
