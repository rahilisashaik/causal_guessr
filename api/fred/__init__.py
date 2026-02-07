"""FRED (Federal Reserve Economic Data) API client."""

from api.fred.client import (
    get_observations,
    get_observations_cached,
    get_series,
)

__all__ = [
    "get_observations",
    "get_observations_cached",
    "get_series",
]
