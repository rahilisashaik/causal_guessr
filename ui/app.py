"""
Minimal web UI: one-time gameplay. GET /api/game/new generates one puzzle (LLM + FRED/Google Trends),
serves chart; POST /api/game/guess checks the guess against the current game (4 attempts, hints).
"""

import base64
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).resolve().parent / "static"

logger = logging.getLogger(__name__)

# Ensure investigation logs are visible (seed source, FRED 403, retries)
for _log in ("api.seed_generator", "api.fred", "ui.app"):
    logging.getLogger(_log).setLevel(logging.INFO)


def _slug(series_id: str, start: str, end: str, index: int) -> str:
    """Unique id for a puzzle from series, dates, and index."""
    return f"fred-{series_id}-{start}-{end}-{index}"


def _slug_trends(keyword: str, start: str, end: str, index: int) -> str:
    """Unique id for a Google Trends puzzle (keyword may contain spaces)."""
    safe = (keyword or "").replace(" ", "_")[:30]
    return f"google_trends-{safe}-{start}-{end}-{index}"


def _slug_nber(series_id: str, start: str, end: str, index: int) -> str:
    """Unique id for an NBER Macrohistory puzzle."""
    safe = (series_id or "").replace("/", "-")[:40]
    return f"nber-{safe}-{start}-{end}-{index}"


def _normalize_guess(guess: str) -> str:
    return guess.strip().lower()


def _check_guess(guess: str, acceptable: list[str]) -> bool:
    g = _normalize_guess(guess)
    return g in [a.strip().lower() for a in acceptable]


# Single current game in memory: set when /api/game/new succeeds; used by /api/game/guess
_current_game: dict | None = None

# Session state: avoid repeating time intervals or y-axis metrics per server lifetime
_session_displayed_intervals: list[tuple[str, str]] = []
_session_displayed_metrics: set[str] = set()

# Seed produced by GET /api/game/seed; consumed by POST /api/game/build for stage-based loading UI
_pending_seed: dict | None = None


def _intervals_overlap(start: str, end: str, intervals: list[tuple[str, str]]) -> bool:
    """True if (start, end) overlaps any (a_start, a_end) in intervals. YYYY-MM-DD string comparison."""
    for a_start, a_end in intervals:
        if start <= a_end and a_start <= end:
            return True
    return False


def _metric_key(seed: dict) -> str:
    """Canonical key for the y-axis metric: fred:seriesId, google_trends:searchTerm (normalized), nber:seriesId."""
    source = (seed.get("source") or "fred").strip().lower()
    if source == "google_trends":
        term = (seed.get("searchTerm") or "").strip().lower()
        return f"google_trends:{term}"
    if source == "nber":
        return f"nber:{seed.get('seriesId') or ''}"
    return f"fred:{seed.get('seriesId') or ''}"


app = FastAPI(title="Causal Guessr")


@app.on_event("startup")
def startup_cache_fred_releases():
    """Cache FRED releases at startup for release-based seed discovery."""
    try:
        from api.fred import get_releases_cached
        get_releases_cached()
    except Exception as e:
        logger.warning("Startup FRED releases cache failed (release discovery may be empty): %s", e)


@app.get("/api/debug/openai")
def debug_openai_key():
    """
    Verify OPENAI_API_KEY is loaded. Returns configured=true/false and a safe key prefix.
    Do not use in production if you want to hide whether a key is set.
    """
    import os
    raw = os.environ.get("OPENAI_API_KEY")
    key = (raw or "").strip().strip('"').strip("'") if raw else None
    return {
        "configured": bool(key),
        "key_prefix": (key[:12] + "…") if key and len(key) > 12 else ("…" if key else None),
        "key_length": len(key) if key else 0,
    }


_MAX_SEED_RETRIES = 5  # try up to this many seeds when FRED/Trends fails


def _resolve_fred_discovery(seed: dict) -> None:
    """
    If seed has fredDiscovery "search" or "release", resolve to a seriesId (mutates seed).
    Raises ValueError if discovery params missing or no series found for the date range.
    """
    discovery = (seed.get("fredDiscovery") or "").strip().lower()
    if not discovery:
        return
    start = seed.get("startDate") or ""
    end = seed.get("endDate") or ""
    if not start or not end:
        raise ValueError("FRED discovery seed missing startDate or endDate")
    series_list: list[dict] = []
    if discovery == "search":
        search_text = (seed.get("searchText") or "").strip()
        if not search_text:
            raise ValueError("FRED discovery=search missing searchText")
        from api.fred import search_series
        series_list = search_series(search_text, limit=50)
    elif discovery == "release":
        try:
            release_id = int(seed.get("releaseId") or seed.get("release_id") or 0)
        except (TypeError, ValueError):
            raise ValueError("FRED discovery=release missing or invalid releaseId")
        if release_id <= 0:
            raise ValueError("FRED discovery=release missing releaseId")
        from api.fred import get_release_series
        series_list = get_release_series(release_id, limit=500)
    else:
        return
    # Keep series that cover [start, end]: observation_start <= end and observation_end >= start
    valid = [
        s for s in series_list
        if (s.get("observation_start") or "") <= end and (s.get("observation_end") or "0000") >= start
    ]
    if not valid:
        raise ValueError(
            f"No FRED series found for {discovery} covering {start} to {end}"
        )
    # Prefer by popularity (desc), then take first
    valid.sort(key=lambda s: -(s.get("popularity") or 0))
    chosen = valid[0]
    seed["seriesId"] = chosen.get("id") or chosen.get("series_id")
    if not seed["seriesId"]:
        raise ValueError("FRED series object missing id")


def _get_valid_seed(user_pref: str | None) -> dict:
    """Return a seed that passes session overlap and metric checks. Raises HTTPException after max retries."""
    from api.seed_generator import generate_puzzle_seed

    for attempt in range(1, _MAX_SEED_RETRIES + 1):
        try:
            seed = generate_puzzle_seed(user_preference=user_pref)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Could not generate puzzle seed: {e!s}",
            ) from e
        start = seed["startDate"]
        end = seed["endDate"]
        if _intervals_overlap(start, end, _session_displayed_intervals):
            continue
        if _metric_key(seed) in _session_displayed_metrics:
            continue
        return seed
    raise HTTPException(
        status_code=503,
        detail="Could not generate a puzzle seed that passes session diversity checks after max retries.",
    )


def _metadata_and_title_from_seed(seed: dict) -> tuple[dict, str, str]:
    """Build puzzle metadata and title from a seed. Returns (metadata, title, log_id). Raises on invalid seed."""
    from api.fred import get_series

    source = seed.get("source", "fred")
    start = seed["startDate"]
    end = seed["endDate"]

    if source == "google_trends":
        keyword = seed.get("searchTerm") or ""
        if not keyword:
            raise ValueError("google_trends seed missing searchTerm")
        pid = _slug_trends(keyword, start, end, 0)
        title = f"Search: {keyword}"
        metadata = {
            "id": pid,
            "source": "google_trends",
            "title": title,
            "correctEvent": seed["correctEvent"],
            "acceptableAnswers": seed.get("acceptableAnswers") or [],
            "explanation": seed.get("explanation") or "",
            "data": {
                "searchTerm": keyword,
                "startDate": start,
                "endDate": end,
                "geo": seed.get("geo") or "",
            },
        }
        return metadata, title, keyword
    if source == "nber":
        series_id = seed.get("seriesId") or ""
        if not series_id:
            raise ValueError("nber seed missing seriesId")
        pid = _slug_nber(series_id, start, end, 0)
        try:
            from api.nber import get_series_info
            info = get_series_info(series_id)
            title = (info.get("description") or "").strip() or f"NBER: {series_id}"
        except Exception:
            title = f"NBER: {series_id}"
        metadata = {
            "id": pid,
            "source": "nber",
            "title": title,
            "correctEvent": seed["correctEvent"],
            "acceptableAnswers": seed.get("acceptableAnswers") or [],
            "explanation": seed.get("explanation") or "",
            "data": {"seriesId": series_id, "startDate": start, "endDate": end},
        }
        return metadata, title, series_id
    # fred
    series_id = seed.get("seriesId") or ""
    if not series_id:
        raise ValueError("fred seed missing seriesId")
    pid = _slug(series_id, start, end, 0)
    title = series_id
    try:
        info = get_series(series_id)
        title = info.get("title", series_id)
    except Exception:
        pass
    metadata = {
        "id": pid,
        "source": "fred",
        "title": title,
        "correctEvent": seed["correctEvent"],
        "acceptableAnswers": seed.get("acceptableAnswers") or [],
        "explanation": seed.get("explanation") or "",
        "data": {"seriesId": series_id, "startDate": start, "endDate": end},
    }
    return metadata, title, series_id


@app.get("/api/game/seed")
def get_seed(preference: str | None = None):
    """
    Generate a puzzle seed that passes session diversity checks and store it for POST /api/game/build.
    Frontend can show "Generating puzzle seed..." during this call.
    Returns { "status": "ok" }. On failure returns 503.
    """
    global _pending_seed
    try:
        _pending_seed = _get_valid_seed((preference or "").strip() or None)
    except HTTPException:
        raise
    return {"status": "ok"}


class BuildBody(BaseModel):
    preference: str | None = None


@app.post("/api/game/build")
def build_game(body: BuildBody | None = None):
    """
    Build the puzzle from the seed stored by GET /api/game/seed (fetch data, render chart), set current game.
    Frontend can show "Fetching economic data..." during this call.
    Call GET /api/game/seed first. Optional body.preference used when retrying with a new seed on fetch failure.
    Returns { id, title, imageBase64, seed_source, attempts_left }. On failure returns 503.
    """
    global _current_game, _pending_seed, _session_displayed_intervals, _session_displayed_metrics

    try:
        from puzzles_factory import build_puzzle
        from visualization.plotter import plot_to_bytes
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Setup error: {e!s}") from e

    user_pref = (body.preference if body else None) or ""
    user_pref = (user_pref or "").strip() or None
    last_error = None

    for attempt in range(1, _MAX_SEED_RETRIES + 1):
        seed = _pending_seed
        if seed is None:
            if attempt == 1:
                raise HTTPException(
                    status_code=409,
                    detail="No pending seed. Call GET /api/game/seed first.",
                )
            try:
                seed = _get_valid_seed(user_pref)
                _pending_seed = seed
            except HTTPException:
                raise
        source = seed.get("source", "fred")
        start = seed["startDate"]
        end = seed["endDate"]
        if _intervals_overlap(start, end, _session_displayed_intervals) or _metric_key(seed) in _session_displayed_metrics:
            _pending_seed = None
            if attempt < _MAX_SEED_RETRIES:
                try:
                    seed = _get_valid_seed(user_pref)
                    _pending_seed = seed
                except HTTPException:
                    raise
                continue
            raise HTTPException(status_code=503, detail="Seed no longer valid for session. Call GET /api/game/seed again.")
        if (source or "fred").strip().lower() == "fred" and seed.get("fredDiscovery"):
            try:
                _resolve_fred_discovery(seed)
            except ValueError as e:
                last_error = e
                _pending_seed = None
                logger.warning("build_game attempt=%s FRED discovery resolve failed: %s", attempt, e)
                if attempt < _MAX_SEED_RETRIES:
                    try:
                        seed = _get_valid_seed(user_pref)
                        _pending_seed = seed
                    except HTTPException:
                        raise
                continue
        try:
            metadata, title, log_id = _metadata_and_title_from_seed(seed)
        except ValueError as e:
            last_error = e
            _pending_seed = None
            logger.warning("build_game attempt=%s metadata from seed failed: %s", attempt, e)
            continue
        logger.info(
            "build_game attempt=%s/%s source=%s id=%s startDate=%s endDate=%s",
            attempt, _MAX_SEED_RETRIES, source, log_id, start, end,
        )
        try:
            puzzle = build_puzzle(metadata)
            png_bytes = plot_to_bytes(puzzle)
        except Exception as e:
            last_error = e
            _pending_seed = None
            logger.warning("build_game attempt=%s/%s failed source=%s id=%s error=%s", attempt, _MAX_SEED_RETRIES, source, log_id, e)
            if attempt < _MAX_SEED_RETRIES:
                try:
                    seed = _get_valid_seed(user_pref)
                    _pending_seed = seed
                except HTTPException:
                    raise
            continue
        image_b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        seed_source = seed.get("seed_source", "unknown")
        hints = list(seed.get("hints") or [])[:4]
        while len(hints) < 4:
            hints.append(seed.get("correctEvent") or "The correct event.")
        _current_game = {
            "id": metadata["id"],
            "title": title,
            "imageBase64": image_b64,
            "correctEvent": seed["correctEvent"],
            "acceptableAnswers": list(seed.get("acceptableAnswers") or []),
            "explanation": seed.get("explanation") or "",
            "seed_source": seed_source,
            "hints": hints,
            "attempts_left": 4,
        }
        _session_displayed_intervals.append((start, end))
        _session_displayed_metrics.add(_metric_key(seed))
        _pending_seed = None
        return {
            "id": _current_game["id"],
            "title": _current_game["title"],
            "imageBase64": _current_game["imageBase64"],
            "seed_source": seed_source,
            "attempts_left": 4,
        }
    raise HTTPException(
        status_code=503,
        detail=f"Could not fetch or render chart after {_MAX_SEED_RETRIES} tries. Last error: {last_error!s}",
    )


@app.get("/api/game/new")
def new_game(preference: str | None = None):
    """
    One-shot: generate seed and build puzzle (same as GET /api/game/seed then POST /api/game/build).
    Kept for backward compatibility. Prefer the two-step flow for stage-based loading messages.
    """
    global _pending_seed
    get_seed(preference)
    return build_game(BuildBody(preference=preference))


class GuessBody(BaseModel):
    guess: str


@app.post("/api/game/guess")
def submit_guess(body: GuessBody):
    """
    Check guess against the current game. Player has 4 attempts; each wrong guess
    returns the next hint. Answer (correctEvent) only returned when attempts exhausted.
    """
    if _current_game is None:
        raise HTTPException(
            status_code=409,
            detail="No active game. Call GET /api/game/new first.",
        )

    attempts_left = max(0, _current_game.get("attempts_left", 0))
    if attempts_left <= 0:
        return {
            "correct": False,
            "attempts_left": 0,
            "hint": None,
            "correctEvent": _current_game.get("correctEvent") or "",
            "explanation": _current_game.get("explanation") or "",
        }

    acceptable = list(_current_game.get("acceptableAnswers") or [])
    correct_event = _current_game.get("correctEvent") or ""
    if correct_event:
        acceptable.append(correct_event)
    correct = _check_guess(body.guess, acceptable)

    # If not in the list, ask LLM whether the guess is semantically correct
    if not correct:
        try:
            from api.guess_evaluator import evaluate_guess_with_llm
            correct = evaluate_guess_with_llm(
                body.guess,
                correct_event,
                list(_current_game.get("acceptableAnswers") or []),
            )
        except Exception:
            correct = False

    if correct:
        return {
            "correct": True,
            "attempts_left": attempts_left,
            "explanation": _current_game.get("explanation") or "",
        }

    # Single decrement per request; never go below 0
    _current_game["attempts_left"] = max(0, attempts_left - 1)
    attempts_left = _current_game["attempts_left"]
    hints = _current_game.get("hints") or []
    hint_index = 4 - attempts_left - 1
    hint = hints[hint_index] if 0 <= hint_index < len(hints) else None

    out = {
        "correct": False,
        "attempts_left": attempts_left,
        "hint": hint,
    }
    # Always include answer when no attempts left so the UI can show it
    if attempts_left <= 0:
        out["correctEvent"] = _current_game.get("correctEvent") or ""
        out["explanation"] = _current_game.get("explanation") or ""
    return out


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")
