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
    },
    {
        "seriesId": "FEDFUNDS",
        "startDate": "2007-01-01",
        "endDate": "2009-12-31",
        "correctEvent": "2008 financial crisis",
        "acceptableAnswers": ["2008", "financial crisis", "recession", "fed", "rate cuts"],
        "explanation": "The Fed cut rates aggressively in response to the 2008 crisis.",
    },
]


def _random_seed_from_file() -> dict:
    """Return a random puzzle seed from puzzle_seeds.json. Raises if file missing or empty."""
    if not PUZZLE_SEEDS_PATH.exists():
        raise FileNotFoundError("puzzle_seeds.json not found")
    with open(PUZZLE_SEEDS_PATH) as f:
        seeds = json.load(f)
    if not seeds:
        raise ValueError("puzzle_seeds.json is empty")
    seed = random.choice(seeds)
    if not isinstance(seed.get("acceptableAnswers"), list):
        seed["acceptableAnswers"] = [seed["acceptableAnswers"]] if seed.get("acceptableAnswers") else []
    logger.info(
        "seed_source=fallback seriesId=%s startDate=%s endDate=%s correctEvent=%s",
        seed.get("seriesId"),
        seed.get("startDate"),
        seed.get("endDate"),
        seed.get("correctEvent"),
    )
    return seed


def generate_puzzle_seed() -> dict:
    """
    Produce one puzzle seed: try LLM (OpenAI) first; on failure (quota, no key, etc.)
    fall back to a random seed from puzzle_seeds.json.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return _random_seed_from_file()

    raw_key = os.environ.get("OPENAI_API_KEY")
    api_key = (raw_key or "").strip().strip('"').strip("'") if raw_key else None
    if not api_key:
        logger.info("OPENAI_API_KEY missing or empty, using fallback seed")
        return _random_seed_from_file()

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

        prompt = f"""You are helping create a single "causal guessr" puzzle. The puzzle shows a time-series chart from FRED (Federal Reserve Economic Data). The player must guess what real-world event caused the visible trend.

Output exactly one puzzle seed as a JSON object with these keys (no other text, no markdown):
- seriesId: a FRED series ID from this list (use only these): {series_list}
- startDate: YYYY-MM-DD (start of the date range to fetch)
- endDate: YYYY-MM-DD (end of the date range)
- correctEvent: short name of the causal event (e.g. "COVID-19 pandemic", "2008 financial crisis")
- acceptableAnswers: list of strings that count as correct (lowercase variants, e.g. "covid", "covid-19", "pandemic")
- explanation: one sentence explaining why this event caused this trend

Examples of valid seeds:
{examples_str}

Pick a series and date range that has a clear, famous causal story (recession, pandemic, policy, etc.). Vary the event and series; do not always pick COVID or 2008. Output only the JSON object for one puzzle seed."""

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
        required = ["seriesId", "startDate", "endDate", "correctEvent", "acceptableAnswers", "explanation"]
        for k in required:
            if k not in seed:
                raise ValueError(f"LLM seed missing key: {k}")
        if not isinstance(seed["acceptableAnswers"], list):
            seed["acceptableAnswers"] = [seed["acceptableAnswers"]]
        logger.info(
            "seed_source=llm seriesId=%s startDate=%s endDate=%s correctEvent=%s",
            seed.get("seriesId"),
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
            return _random_seed_from_file()
        # 401/403: re-raise so you know the key is invalid
        if "401" in err_str or "403" in err_str:
            logger.warning("LLM seed generation failed (auth): %s", e)
            raise
        logger.warning("LLM seed generation failed, using fallback: %s", e, exc_info=False)
        return _random_seed_from_file()
