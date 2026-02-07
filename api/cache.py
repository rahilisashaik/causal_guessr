"""
In-memory cache for API responses (FRED, and later Google Trends).
Key: (source, series_id, start_date, end_date). Value: list of {date, value}.
"""

from typing import Callable

# Module-level store; key = _make_key(...), value = list of observations
_cache: dict[tuple[str, str, str, str], list[dict]] = {}


def _make_key(
    source: str,
    series_id: str,
    start_date: str,
    end_date: str,
) -> tuple[str, str, str, str]:
    """Build cache key from source and series params."""
    return (source, series_id, start_date, end_date)


def _get(key: tuple[str, str, str, str]) -> list[dict] | None:
    """Return cached value if present, else None."""
    return _cache.get(key)


def _set(key: tuple[str, str, str, str], value: list[dict]) -> None:
    """Store value in cache."""
    _cache[key] = value


def get_or_fetch(
    source: str,
    series_id: str,
    start_date: str,
    end_date: str,
    fetcher: Callable[[], list[dict]],
) -> list[dict]:
    """
    Return observations from cache if present; otherwise call fetcher(), store, and return.

    Args:
        source: e.g. "fred".
        series_id: Series or query id (e.g. FRED series_id).
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD.
        fetcher: No-arg callable that returns the observations list (e.g. calls the API).

    Returns:
        List of {"date": "YYYY-MM-DD", "value": "..."}.
    """
    key = _make_key(source, series_id, start_date, end_date)
    cached = _get(key)
    if cached is not None:
        return cached
    value = fetcher()
    _set(key, value)
    return value


def clear() -> None:
    """Clear the cache (e.g. for tests)."""
    _cache.clear()
