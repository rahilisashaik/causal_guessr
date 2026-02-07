"""NBER Macrohistory Database API client."""

from api.nber.client import (
    get_observations,
    get_observations_cached,
    get_series_info,
)

__all__ = [
    "get_observations",
    "get_observations_cached",
    "get_series_info",
]
