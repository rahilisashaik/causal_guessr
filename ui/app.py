"""
Minimal web UI: one-time gameplay. GET /api/game/new generates one puzzle (LLM + FRED),
serves chart; POST /api/game/guess checks the guess against the current game.
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
for _log in ("api.seed_generator", "api.fred_client", "ui.app"):
    logging.getLogger(_log).setLevel(logging.INFO)


def _slug(series_id: str, start: str, end: str, index: int) -> str:
    """Unique id for a puzzle from series, dates, and index."""
    return f"fred-{series_id}-{start}-{end}-{index}"


def _normalize_guess(guess: str) -> str:
    return guess.strip().lower()


def _check_guess(guess: str, acceptable: list[str]) -> bool:
    g = _normalize_guess(guess)
    return g in [a.strip().lower() for a in acceptable]


# Single current game in memory: set when /api/game/new succeeds; used by /api/game/guess
_current_game: dict | None = None


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


_MAX_SEED_RETRIES = 5  # try up to this many seeds when FRED returns 403 or other data errors


@app.get("/api/game/new")
def new_game():
    """
    Generate one puzzle via LLM (seed) + FRED (data), render chart, set as current game.
    If FRED returns 403 for a series, retries with a new seed (up to _MAX_SEED_RETRIES).
    Returns { id, title, imageBase64 }. On failure returns 503.
    """
    global _current_game

    try:
        from api.seed_generator import generate_puzzle_seed
        from api.fred_client import get_series
        from puzzles_factory import build_puzzle
        from visualization.plotter import plot_to_bytes
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Setup error: {e!s}") from e

    last_error = None
    for attempt in range(1, _MAX_SEED_RETRIES + 1):
        try:
            seed = generate_puzzle_seed()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Could not generate puzzle seed: {e!s}",
            ) from e

        series_id = seed["seriesId"]
        start = seed["startDate"]
        end = seed["endDate"]
        pid = _slug(series_id, start, end, 0)
        logger.info(
            "new_game attempt=%s/%s seriesId=%s startDate=%s endDate=%s",
            attempt,
            _MAX_SEED_RETRIES,
            series_id,
            start,
            end,
        )

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

        try:
            puzzle = build_puzzle(metadata)
            png_bytes = plot_to_bytes(puzzle)
        except Exception as e:
            last_error = e
            logger.warning(
                "new_game attempt=%s/%s failed seriesId=%s startDate=%s endDate=%s error=%s",
                attempt,
                _MAX_SEED_RETRIES,
                series_id,
                start,
                end,
                e,
            )
            continue

        image_b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        _current_game = {
            "id": pid,
            "title": title,
            "imageBase64": image_b64,
            "correctEvent": seed["correctEvent"],
            "acceptableAnswers": list(seed.get("acceptableAnswers") or []),
            "explanation": seed.get("explanation") or "",
        }

        return {
            "id": _current_game["id"],
            "title": _current_game["title"],
            "imageBase64": _current_game["imageBase64"],
        }

    raise HTTPException(
        status_code=503,
        detail=f"Could not fetch or render chart after {_MAX_SEED_RETRIES} tries (FRED may be returning 403 for many series). Last error: {last_error!s}",
    )


class GuessBody(BaseModel):
    guess: str


@app.post("/api/game/guess")
def submit_guess(body: GuessBody):
    """Check guess against the current game. Returns { correct, explanation, correctEvent }."""
    if _current_game is None:
        raise HTTPException(
            status_code=409,
            detail="No active game. Call GET /api/game/new first.",
        )

    acceptable = list(_current_game.get("acceptableAnswers") or [])
    if _current_game.get("correctEvent"):
        acceptable.append(_current_game["correctEvent"])
    correct = _check_guess(body.guess, acceptable)

    return {
        "correct": correct,
        "explanation": _current_game.get("explanation", ""),
        "correctEvent": _current_game.get("correctEvent", ""),
    }


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")
