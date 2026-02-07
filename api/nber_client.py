"""
NBER Macrohistory Database client. Fetches .db series from data.nber.org,
parses annual/quarterly/monthly data, and returns observations as {date, value}.
No API key required. Data covers 1800s–1940s (pre-WWI and interwar).
"""

import logging

import requests

logger = logging.getLogger(__name__)

NBER_BASE = "https://data.nber.org/databases/macrohistory/data"


def _parse_db_content(content: str) -> list[dict]:
    """
    Parse NBER .db format: comment lines, then -1 (annual) / -4 (quarterly) / -12 (monthly),
    then start/end lines (e.g. 1862. / 1930.), then one value per line. NA = missing.
    Returns list of {"date": "YYYY-MM-DD", "value": str}.
    """
    lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
    # Find frequency and start/end
    freq = None
    start_line = None
    for i, ln in enumerate(lines):
        if ln in ("-1", "-4", "-12"):
            freq = int(ln)
            if i + 2 < len(lines):
                start_line = i + 1
            break
        if ln.startswith('"'):
            continue
        try:
            int(ln)
            break
        except ValueError:
            continue

    if freq is None or start_line is None:
        return []

    start_str = lines[start_line].rstrip(".")
    end_str = lines[start_line + 1].rstrip(".")
    try:
        start_y = int(float(start_str))
        end_y = int(float(end_str))
        # Subperiod only for quarterly/monthly (e.g. 1930.25 = Q2)
        start_sub = 1
        end_sub = 1
        if abs(freq) == 4:
            end_sub = 4
        elif abs(freq) == 12:
            end_sub = 12
        if "." in lines[start_line]:
            start_sub = max(1, min(int(round((float(start_str) % 1) * abs(freq))) or 1, abs(freq)))
        if "." in lines[start_line + 1] and abs(freq) in (4, 12):
            end_sub = max(1, min(int(round((float(end_str) % 1) * abs(freq))) or abs(freq), abs(freq)))
    except (ValueError, IndexError):
        return []

    value_lines = lines[start_line + 2 :]
    obs = []
    y, sub = start_y, start_sub
    for v in value_lines:
        is_na = (v or "").strip().upper() == "NA" or not (v or "").strip()
        if freq == -1:
            date_str = f"{y}-01-01"
            obs.append({"date": date_str, "value": v.strip() if not is_na else "NA"})
            y += 1
        elif freq == -4:
            month = (sub - 1) * 3 + 1
            date_str = f"{y}-{month:02d}-01"
            obs.append({"date": date_str, "value": v.strip() if not is_na else "NA"})
            sub += 1
            if sub > 4:
                sub = 1
                y += 1
        else:  # -12 monthly
            date_str = f"{y}-{sub:02d}-01"
            obs.append({"date": date_str, "value": v.strip() if not is_na else "NA"})
            sub += 1
            if sub > 12:
                sub = 1
                y += 1
        if y > end_y or (y == end_y and ((freq == -12 and sub > end_sub) or (freq == -4 and sub > end_sub))):
            break
    return obs


def get_observations(
    series_id: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch NBER Macrohistory series and return observations in range.
    series_id: path like "01/a01005a" (chapter/filename without .db).
    start_date, end_date: YYYY-MM-DD.
    Returns list of {"date": "YYYY-MM-DD", "value": "..."}.
    """
    url = f"{NBER_BASE}/{series_id}.db"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("NBER fetch failed for %s: %s", series_id, e)
        raise

    raw = resp.text
    all_obs = _parse_db_content(raw)
    if not all_obs:
        return []

    out = [o for o in all_obs if start_date <= o["date"] <= end_date]
    logger.info("NBER series_id=%s start=%s end=%s n_obs=%s", series_id, start_date, end_date, len(out))
    return out


def get_observations_cached(
    series_id: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Same as get_observations but uses in-memory cache."""
    from api.cache import get_or_fetch

    def _fetch() -> list[dict]:
        return get_observations(series_id, start_date, end_date)

    return get_or_fetch("nber", series_id, start_date, end_date, _fetch)


if __name__ == "__main__":
    # Self-test: 63 values + 6 NA at end → still 69 obs, last date 1930-01-01
    num_years = 1930 - 1862 + 1  # 69
    vals = [82.2 + (i * 0.5) for i in range(63)]  # 63 numeric
    content = '" Index of crop production\n-1\n1862.\n1930.\n' + "\n".join(str(v) for v in vals) + "\nNA\nNA\nNA\nNA\nNA\nNA"
    obs = _parse_db_content(content)
    assert len(obs) == num_years, f"expected {num_years} obs, got {len(obs)}"
    assert obs[0]["date"] == "1862-01-01", obs[0]["date"]
    assert obs[-1]["date"] == "1930-01-01", obs[-1]["date"]
    assert obs[-1]["value"] == "NA", obs[-1]["value"]
    print("parse ok: first", obs[0]["date"], "last", obs[-1]["date"], "count", len(obs))
