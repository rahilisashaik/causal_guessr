"""
Generate a single puzzle seed using an LLM (OpenAI). Used for one-time gameplay:
the LLM picks a FRED series and date range plus the causal event and acceptable answers.
Falls back to a random seed from puzzle_seeds.json if the LLM fails (e.g. quota, no key).
"""

import json
import logging
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUZZLE_SEEDS_PATH = PROJECT_ROOT / "puzzle_seeds.json"

# Known FRED series that have clear causal stories (unemployment, GDP, rates, etc.)
FRED_SERIES_EXAMPLES = [
    "UNRATE", "ICSA", "GDPC1", "GDP", "INDPRO", "RSXFS", "PAYEMS", "HOUST",
    "CIVPART", "FEDFUNDS", "CPIAUCSL", "M2SL", "DSPIC96", "T10Y2Y", "TOTALSA",
    "PERMIT", "DCOILWTICO", "VIXCLS", "BAMLH0A0HYM2", "TEDRATE", "UMCSENT",
    "TCU", "CSUSHPISA", "PSAVERT", "MORTGAGE30US", "GDI", "A191RL1Q225SBEA",
]

FEW_SHOT_SEEDS = [
    {
        "seriesId": "UNRATE",
        "startDate": "2020-01-01",
        "endDate": "2020-12-31",
        "correctEvent": "COVID-19 pandemic",
        "acceptableAnswers": ["covid", "covid-19", "coronavirus", "pandemic"],
        "explanation": "Lockdowns and layoffs in spring 2020 caused a sharp rise in unemployment.",
        "hints": [
            "Think about a major global disruption that began in early 2020.",
            "It led to widespread lockdowns and a sharp drop in economic activity.",
            "The event is often referred to by a short acronym or number.",
            "COVID-19 pandemic",
        ],
    },
    {
        "seriesId": "FEDFUNDS",
        "startDate": "2007-01-01",
        "endDate": "2009-12-31",
        "correctEvent": "2008 financial crisis",
        "acceptableAnswers": ["2008", "financial crisis", "recession", "fed", "rate cuts"],
        "explanation": "The Fed cut rates aggressively in response to the 2008 crisis.",
        "hints": [
            "Consider a major financial shock that peaked around 2008.",
            "Central banks responded by cutting interest rates sharply.",
            "It is often called the Great Recession or named after a year.",
            "2008 financial crisis",
        ],
    },
    {
        "source": "google_trends",
        "searchTerm": "toilet paper",
        "startDate": "2020-01-01",
        "endDate": "2020-06-30",
        "correctEvent": "COVID-19 pandemic",
        "acceptableAnswers": ["covid", "covid-19", "coronavirus", "pandemic", "panic buying"],
        "explanation": "Panic buying and stockpiling in early 2020 caused a spike in search interest for toilet paper.",
        "hints": [
            "Think about what people suddenly searched for in early 2020.",
            "Shortages and stockpiling drove interest in this everyday product.",
            "The event was a global health crisis with lockdowns.",
            "COVID-19 pandemic",
        ],
    },
]


def _ensure_hints(seed: dict) -> dict:
    """Ensure seed has exactly 4 hints (increasingly obvious). Build from explanation/correctEvent if missing."""
    hints = seed.get("hints")
    if isinstance(hints, list) and len(hints) >= 4:
        seed["hints"] = hints[:4]
        return seed
    expl = seed.get("explanation") or ""
    event = seed.get("correctEvent") or "the correct event"
    seed["hints"] = [
        "Think about major economic or global events in this date range.",
        expl or "The trend is linked to a well-known historical event.",
        "The answer is often abbreviated or has a common short name.",
        event,
    ]
    return seed


def _is_2019_2021(start: str, end: str) -> bool:
    """True if date range overlaps 2019-2021."""
    if not start or not end:
        return False
    return start <= "2021-12-31" and end >= "2019-01-01"


def _is_covid_event(correct_event: str) -> bool:
    """True if correctEvent is COVID-related."""
    return "covid" in (correct_event or "").lower() or "pandemic" in (correct_event or "").lower()


def _random_seed_from_file(*, avoid_2019_2021_and_covid: bool = False) -> dict:
    """Return a random puzzle seed from puzzle_seeds.json. Raises if file missing or empty."""
    if not PUZZLE_SEEDS_PATH.exists():
        raise FileNotFoundError("puzzle_seeds.json not found")
    with open(PUZZLE_SEEDS_PATH) as f:
        seeds = json.load(f)
    if not seeds:
        raise ValueError("puzzle_seeds.json is empty")
    if avoid_2019_2021_and_covid:
        preferred = [
            s for s in seeds
            if not (_is_2019_2021(s.get("startDate") or "", s.get("endDate") or "")
                    or _is_covid_event(s.get("correctEvent")))
        ]
        if preferred:
            seeds = preferred
    seed = random.choice(seeds)
    if not isinstance(seed.get("acceptableAnswers"), list):
        seed["acceptableAnswers"] = [seed["acceptableAnswers"]] if seed.get("acceptableAnswers") else []
    seed["seed_source"] = "fallback"
    _ensure_hints(seed)
    logger.info(
        "seed_source=fallback seriesId=%s startDate=%s endDate=%s correctEvent=%s",
        seed.get("seriesId"),
        seed.get("startDate"),
        seed.get("endDate"),
        seed.get("correctEvent"),
    )
    return seed


def generate_puzzle_seed(
    *,
    session_2019_2021_count: int = 0,
    session_covid_count: int = 0,
) -> dict:
    """
    Produce one puzzle seed: try LLM (OpenAI) first; on failure (quota, no key, etc.)
    fall back to a random seed from puzzle_seeds.json.
    When session_2019_2021_count or session_covid_count >= 1, bias away from 2019-2021 and COVID.
    """
    avoid_2019_2021_covid = session_2019_2021_count >= 1 or session_covid_count >= 1

    try:
        from openai import OpenAI
    except ImportError:
        return _random_seed_from_file(avoid_2019_2021_and_covid=avoid_2019_2021_covid)

    raw_key = os.environ.get("OPENAI_API_KEY")
    api_key = (raw_key or "").strip().strip('"').strip("'") if raw_key else None
    if not api_key:
        logger.info("OPENAI_API_KEY missing or empty, using fallback seed")
        return _random_seed_from_file(avoid_2019_2021_and_covid=avoid_2019_2021_covid)

    # Safe hint for verification (never log the full key)
    key_hint = (api_key[:12] + "…") if len(api_key) > 12 else "…"
    logger.info(
        "OPENAI_API_KEY is set (prefix=%s), model=%s",
        key_hint,
        os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    )
    try:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        examples_str = json.dumps(FEW_SHOT_SEEDS, indent=2)
        series_list = ", ".join(FRED_SERIES_EXAMPLES)

        # Randomly request one source so we get variety (avoid always getting the same example)
        requested_source = random.choice(["fred", "google_trends"])

        if requested_source == "fred":
            source_instruction = f'''You MUST output a FRED seed with "source": "fred".
- seriesId: from this list ONLY: {series_list}
- startDate, endDate: YYYY-MM-DD
- correctEvent, acceptableAnswers (list), explanation, hints (array of 4 strings, increasingly obvious)
Pick a DIFFERENT series and date range than the examples (e.g. different recession, different event).'''
        else:
            source_instruction = '''You MUST output a Google Trends seed with "source": "google_trends".
- searchTerm: a phrase people actually search (e.g. "face mask", "stimulus check", "vaccine", "inflation")
- startDate, endDate: YYYY-MM-DD
- correctEvent, acceptableAnswers (list), explanation, hints (array of 4 strings, increasingly obvious)
Pick a DIFFERENT search term and event than the examples (not COVID).'''

        avoid_instruction = ""
        if avoid_2019_2021_covid:
            avoid_instruction = """

Important: This session has already had puzzles from 2019-2021 or COVID-19. You MUST NOT use date range 2019-2021 (use e.g. 2007-2009, 2001-2003, 1980-1983) and you MUST NOT use COVID-19 pandemic as the correctEvent (use e.g. 2008 financial crisis, dot-com bust, early 1980s recession, oil crisis)."""

        prompt = f"""You are helping create a single "causal guessr" puzzle. The puzzle shows a time-series chart. The player has 4 guesses; after each wrong guess they get the next hint. Do not give away the answer until the 4th hint.

{source_instruction}{avoid_instruction}

Examples (for format only; do not copy):
{examples_str}

Output exactly one JSON object (no other text, no markdown)."""

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text = (resp.choices[0].message.content or "").strip()

        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()

        seed = json.loads(text)
        source = (seed.get("source") or "fred").strip().lower()
        if source == "google_trends":
            required = ["searchTerm", "startDate", "endDate", "correctEvent", "acceptableAnswers", "explanation"]
        else:
            source = "fred"
            seed["source"] = "fred"
            required = ["seriesId", "startDate", "endDate", "correctEvent", "acceptableAnswers", "explanation"]
        for k in required:
            if k not in seed:
                raise ValueError(f"LLM seed missing key: {k}")
        if not isinstance(seed["acceptableAnswers"], list):
            seed["acceptableAnswers"] = [seed["acceptableAnswers"]]
        _ensure_hints(seed)
        seed["seed_source"] = "llm"
        logger.info(
            "seed_source=llm source=%s id=%s startDate=%s endDate=%s correctEvent=%s",
            source,
            seed.get("searchTerm") or seed.get("seriesId"),
            seed.get("startDate"),
            seed.get("endDate"),
            seed.get("correctEvent"),
        )
        return seed
    except Exception as e:
        err_str = str(e).lower()
        # 429 quota: use fallback so game still works; log clearly so you can fix billing
        if "429" in err_str or "quota" in err_str or "insufficient_quota" in err_str:
            logger.warning(
                "OpenAI returned 429 (insufficient quota). Using fallback seed. "
                "To use LLM-generated seeds: add a payment method at https://platform.openai.com/account/billing"
            )
            return _random_seed_from_file(avoid_2019_2021_and_covid=avoid_2019_2021_covid)
        # 401/403: re-raise so you know the key is invalid
        if "401" in err_str or "403" in err_str:
            logger.warning("LLM seed generation failed (auth): %s", e)
            raise
        logger.warning("LLM seed generation failed, using fallback: %s", e, exc_info=False)
        return _random_seed_from_file(avoid_2019_2021_and_covid=avoid_2019_2021_covid)
