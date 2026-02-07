"""Google Trends API client (pytrends)."""

from api.google_trends.client import (
    get_interest_over_time,
    get_interest_over_time_cached,
)

__all__ = [
    "get_interest_over_time",
    "get_interest_over_time_cached",
]
