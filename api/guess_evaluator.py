"""
Use an LLM to evaluate whether a user's guess is semantically correct when it does not
exactly match the acceptable-answers list. Returns a boolean.
"""

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


def evaluate_guess_with_llm(
    guess: str,
    correct_event: str,
    acceptable_answers: list[str],
) -> bool:
    """
    Ask the LLM whether the guess means the same as the correct answer.
    Use when the guess is not in the acceptable_answers list.
    Returns True only if the LLM says the guess is correct; on API failure returns False.
    """
    guess = (guess or "").strip()
    if not guess:
        return False

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai not installed, cannot evaluate guess with LLM")
        return False

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, cannot evaluate guess with LLM")
        return False

    correct_event = correct_event or ""
    others = [a for a in (acceptable_answers or []) if a and a.strip().lower() != correct_event.strip().lower()]
    others_str = ", ".join(others[:10]) if others else "none"

    from api.prompts import build_guess_evaluation_prompt
    prompt = build_guess_evaluation_prompt(guess, correct_event, others_str)

    try:
        client = OpenAI(api_key=api_key)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip().lower()
    except Exception as e:
        logger.warning("LLM guess evaluation failed: %s", e)
        return False

    # Parse boolean: accept "true", "yes", "1"
    if re.search(r"\btrue\b", text) or re.search(r"\byes\b", text) or text.strip() == "1":
        return True
    return False
