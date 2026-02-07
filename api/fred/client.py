"""
FRED (Federal Reserve Economic Data) API client.
Fetches series observations for a given series_id and date range.
Loads FRED_API_KEY / FRED_API_KEYS from .env in the project root.
Uses round-robin over FRED_API_KEYS when set for parallelized requests.
"""

import logging
import os
import threading
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from project root (api/fred/client.py -> parent.parent.parent = project root)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series"

_keys_list: list[str] | None = None
_keys_lock = threading.Lock()
_key_index = 0


def _get_keys() -> list[str]:
    """Return list of API keys (FRED_API_KEYS comma-separated, or [FRED_API_KEY])."""
    global _keys_list
    with _keys_lock:
        if _keys_list is not None:
            return _keys_list
        raw = os.environ.get("FRED_API_KEYS")
        if raw:
            _keys_list = [k.strip() for k in raw.split(",") if k.strip()]
        else:
            single = os.environ.get("FRED_API_KEY")
            _keys_list = [single] if single else []
        return _keys_list


def _next_key() -> str:
    """Return next API key in round-robin (thread-safe)."""
    global _key_index
    keys = _get_keys()
    if not keys:
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY or FRED_API_KEYS in the environment."
        )
    with _keys_lock:
        key = keys[_key_index % len(keys)]
        _key_index += 1
        return key


def get_observations(
    series_id: str,
    observation_start: str,
    observation_end: str,
    *,
    api_key: str | None = None,
    file_type: str = "json",
) -> list[dict]:
    """
    Fetch observations (time-series data) from FRED for one series.

    Args:
        series_id: FRED series ID (e.g. "UNRATE", "GNPCA").
        observation_start: Start date YYYY-MM-DD.
        observation_end: End date YYYY-MM-DD.
        api_key: FRED API key. If None, uses env var FRED_API_KEY.
        file_type: Response format; "json" is default.

    Returns:
        List of {"date": "YYYY-MM-DD", "value": "..."}. Value may be "." for missing.
    """
    keys = _get_keys()
    if not keys and not api_key:
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY or FRED_API_KEYS in the environment."
        )
    if api_key:
        keys_to_try = [api_key]
    else:
        first = _next_key()
        keys_to_try = [first] + [k for k in keys if k != first]
    last_error = None
    for key_idx, key in enumerate(keys_to_try):
        params = {
            "series_id": series_id,
            "observation_start": observation_start,
            "observation_end": observation_end,
            "api_key": key,
            "file_type": file_type,
        }
        resp = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
        if resp.status_code == 403:
            logger.warning(
                "FRED observations 403 series_id=%s observation_start=%s observation_end=%s key_index=%s/%s (trying Bearer next)",
                series_id,
                observation_start,
                observation_end,
                key_idx + 1,
                len(keys_to_try),
            )
            params.pop("api_key", None)
            resp = requests.get(
                FRED_OBSERVATIONS_URL,
                params=params,
                headers={"Authorization": f"Bearer {key}"},
                timeout=30,
            )
        if resp.status_code == 200:
            data = resp.json()
            observations = data.get("observations", [])
            logger.info(
                "FRED observations 200 series_id=%s observation_start=%s observation_end=%s n_obs=%s",
                series_id,
                observation_start,
                observation_end,
                len(observations),
            )
            return [{"date": ob["date"], "value": ob["value"]} for ob in observations]
        if resp.status_code == 403:
            logger.warning(
                "FRED observations 403 (after Bearer) series_id=%s key_index=%s/%s",
                series_id,
                key_idx + 1,
                len(keys_to_try),
            )
        last_error = resp
    if last_error is not None:
        logger.error(
            "FRED observations failed for all keys series_id=%s observation_start=%s observation_end=%s status=%s",
            series_id,
            observation_start,
            observation_end,
            last_error.status_code,
        )
        logger.error(
            "FRED 403 for all keys usually means the API key(s) do not have API access. "
            "Check https://fred.stlouisfed.org/docs/api/api_key.html and your FRED account."
        )
        last_error.raise_for_status()
    raise ValueError("FRED API key required. Set FRED_API_KEY or FRED_API_KEYS.")


def get_observations_cached(
    series_id: str,
    observation_start: str,
    observation_end: str,
    *,
    api_key: str | None = None,
) -> list[dict]:
    """
    Same as get_observations, but uses the in-memory cache. Repeated calls for the
    same series and date range return cached data without calling the API.
    """
    from api.cache import get_or_fetch

    def _fetch() -> list[dict]:
        return get_observations(
            series_id,
            observation_start,
            observation_end,
            api_key=api_key,
        )

    return get_or_fetch(
        "fred",
        series_id,
        observation_start,
        observation_end,
        _fetch,
    )


def get_series(series_id: str, *, api_key: str | None = None) -> dict:
    """
    Fetch series metadata from FRED (title, observation_start, observation_end, etc.).

    Returns:
        The first (and usually only) series object from the API, e.g. {"id", "title", ...}.
    """
    keys = _get_keys()
    if not keys and not api_key:
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY or FRED_API_KEYS in the environment."
        )
    if api_key:
        keys_to_try = [api_key]
    else:
        first = _next_key()
        keys_to_try = [first] + [k for k in keys if k != first]
    last_error = None
    for key_idx, key in enumerate(keys_to_try):
        params = {"series_id": series_id, "api_key": key, "file_type": "json"}
        resp = requests.get(FRED_SERIES_URL, params=params, timeout=30)
        if resp.status_code == 403:
            logger.warning(
                "FRED series 403 series_id=%s key_index=%s/%s (trying Bearer next)",
                series_id,
                key_idx + 1,
                len(keys_to_try),
            )
            params.pop("api_key", None)
            resp = requests.get(
                FRED_SERIES_URL,
                params=params,
                headers={"Authorization": f"Bearer {key}"},
                timeout=30,
            )
        if resp.status_code == 200:
            data = resp.json()
            seriess = data.get("seriess", [])
            if not seriess:
                raise ValueError(f"No series found for id={series_id!r}")
            return seriess[0]
        if resp.status_code == 403:
            logger.warning("FRED series 403 (after Bearer) series_id=%s", series_id)
        last_error = resp
    if last_error is not None:
        logger.error("FRED series failed for all keys series_id=%s status=%s", series_id, last_error.status_code)
        logger.error(
            "FRED 403 for all keys usually means the API key(s) do not have API access. "
            "Check https://fred.stlouisfed.org/docs/api/api_key.html and your FRED account."
        )
        last_error.raise_for_status()
    raise ValueError("FRED API key required. Set FRED_API_KEY or FRED_API_KEYS.")
