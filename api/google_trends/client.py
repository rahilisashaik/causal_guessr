"""
Google Trends client via pytrends (unofficial, no API key).
Fetches search interest over time for a keyword and date range.
Returns list of {"date": "YYYY-MM-DD", "value": "0-100"} for compatibility with plotter.
"""

import logging

logger = logging.getLogger(__name__)


def get_interest_over_time(
    keyword: str,
    start_date: str,
    end_date: str,
    geo: str = "",
) -> list[dict]:
    """
    Fetch Google Trends interest-over-time for one keyword.

    Args:
        keyword: Search term (e.g. "toilet paper", "COVID").
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD.
        geo: Optional country code (e.g. "" for worldwide, "US" for United States).

    Returns:
        List of {"date": "YYYY-MM-DD", "value": str 0-100}. Empty if no data or error.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        raise ValueError("pytrends required. Install with: pip install pytrends")

    timeframe = f"{start_date} {end_date}"
    try:
        trend = TrendReq(hl="en-US", tz=360)
        trend.build_payload(kw_list=[keyword], timeframe=timeframe, geo=geo if geo else None)
        df = trend.interest_over_time()
    except Exception as e:
        logger.warning("pytrends request failed for keyword=%s: %s", keyword, e)
        raise

    if df is None or df.empty:
        return []

    # Column name is the keyword (possibly normalized); drop 'isPartial' if present
    value_col = None
    for c in df.columns:
        if c != "isPartial":
            value_col = c
            break
    if value_col is None:
        return []

    out = []
    for ts, row in df.iterrows():
        if hasattr(ts, "strftime"):
            date_str = ts.strftime("%Y-%m-%d")
        else:
            date_str = str(ts)[:10]
        val = row.get(value_col)
        if val is None or (isinstance(val, float) and (val != val)):  # NaN
            continue
        out.append({"date": date_str, "value": str(int(val))})
    return out


def get_interest_over_time_cached(
    keyword: str,
    start_date: str,
    end_date: str,
    geo: str = "",
) -> list[dict]:
    """Same as get_interest_over_time but uses in-memory cache."""
    from api.cache import get_or_fetch

    def _fetch() -> list[dict]:
        return get_interest_over_time(keyword, start_date, end_date, geo)

    return get_or_fetch("google_trends", keyword, start_date, end_date, _fetch)
