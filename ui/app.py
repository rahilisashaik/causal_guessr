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

# Session counters: limit 2019-2021 and COVID seeds per server lifetime
_session_2019_2021_count: int = 0
_session_covid_count: int = 0


app = FastAPI(title="Causal Guessr")


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


def _is_2019_2021(start: str, end: str) -> bool:
    return start <= "2021-12-31" and end >= "2019-01-01"


def _is_covid_event(correct_event: str) -> bool:
    return "covid" in (correct_event or "").lower() or "pandemic" in (correct_event or "").lower()


@app.get("/api/game/new")
def new_game():
    """
    Generate one puzzle via LLM (seed) + FRED or Google Trends, render chart, set as current game.
    If fetch fails, retries with a new seed (up to _MAX_SEED_RETRIES).
    Minimizes 2019-2021 and COVID seeds per session (server lifetime).
    Returns { id, title, imageBase64, seed_source, attempts_left }. On failure returns 503.
    """
    global _current_game, _session_2019_2021_count, _session_covid_count

    try:
        from api.seed_generator import generate_puzzle_seed
        from api.fred import get_series
        from puzzles_factory import build_puzzle
        from visualization.plotter import plot_to_bytes
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Setup error: {e!s}") from e

    last_error = None
    for attempt in range(1, _MAX_SEED_RETRIES + 1):
        try:
            seed = generate_puzzle_seed(
                session_2019_2021_count=_session_2019_2021_count,
                session_covid_count=_session_covid_count,
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Could not generate puzzle seed: {e!s}",
            ) from e

        source = seed.get("source", "fred")
        start = seed["startDate"]
        end = seed["endDate"]

        if source == "google_trends":
            keyword = seed.get("searchTerm") or ""
            if not keyword:
                last_error = ValueError("google_trends seed missing searchTerm")
                continue
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
            log_id = keyword
        elif source == "nber":
            series_id = seed.get("seriesId") or ""
            if not series_id:
                last_error = ValueError("nber seed missing seriesId")
                continue
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
                "data": {
                    "seriesId": series_id,
                    "startDate": start,
                    "endDate": end,
                },
            }
            log_id = series_id
        else:
            series_id = seed.get("seriesId") or ""
            if not series_id:
                last_error = ValueError("fred seed missing seriesId")
                continue
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
                "data": {
                    "seriesId": series_id,
                    "startDate": start,
                    "endDate": end,
                },
            }
            log_id = series_id

        logger.info(
            "new_game attempt=%s/%s source=%s id=%s startDate=%s endDate=%s",
            attempt,
            _MAX_SEED_RETRIES,
            source,
            log_id,
            start,
            end,
        )

        try:
            puzzle = build_puzzle(metadata)
            png_bytes = plot_to_bytes(puzzle)
        except Exception as e:
            last_error = e
            logger.warning(
                "new_game attempt=%s/%s failed source=%s id=%s error=%s",
                attempt,
                _MAX_SEED_RETRIES,
                source,
                log_id,
                e,
            )
            continue

        image_b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        seed_source = seed.get("seed_source", "unknown")
        hints = list(seed.get("hints") or [])[:4]
        while len(hints) < 4:
            hints.append(seed.get("correctEvent") or "The correct event.")
        _current_game = {
            "id": pid,
            "title": title,
            "imageBase64": image_b64,
            "correctEvent": seed["correctEvent"],
            "acceptableAnswers": list(seed.get("acceptableAnswers") or []),
            "explanation": seed.get("explanation") or "",
            "seed_source": seed_source,
            "hints": hints,
            "attempts_left": 4,
        }

        # Update session counters to minimize 2019-2021 and COVID in subsequent games
        if _is_2019_2021(start, end):
            _session_2019_2021_count += 1
        if _is_covid_event(seed.get("correctEvent")):
            _session_covid_count += 1

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

    if _current_game.get("attempts_left", 0) <= 0:
        return {
            "correct": False,
            "attempts_left": 0,
            "hint": None,
            "correctEvent": _current_game.get("correctEvent", ""),
            "explanation": _current_game.get("explanation", ""),
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
            "attempts_left": _current_game["attempts_left"],
            "explanation": _current_game.get("explanation", ""),
        }

    _current_game["attempts_left"] = _current_game["attempts_left"] - 1
    attempts_left = _current_game["attempts_left"]
    hints = _current_game.get("hints") or []
    hint_index = 4 - attempts_left - 1
    hint = hints[hint_index] if 0 <= hint_index < len(hints) else None

    out = {
        "correct": False,
        "attempts_left": attempts_left,
        "hint": hint,
    }
    if attempts_left <= 0:
        out["correctEvent"] = _current_game.get("correctEvent", "")
        out["explanation"] = _current_game.get("explanation", "")
    return out


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")
