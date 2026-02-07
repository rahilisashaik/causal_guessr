"""
Centralized prompts for LLM calls. Helper functions build prompt text so we can
edit and tune them in one place; callers pass in dynamic data and use the returned string.
"""

import random


# Era suggestions to rotate and reduce repeated trends (FRED and Google Trends)
ERA_SUGGESTIONS = [
    "Focus on the early 1980s (1980-1983): Volcker recession, inflation fighting.",
    "Focus on the early 1990s (1990-1992): 1990-1991 recession.",
    "Focus on 2000-2003: Dot-com bust, 2001 recession, 9/11 aftermath.",
    "Focus on 2007-2010: 2008 financial crisis, Great Recession, housing crash.",
    "Focus on 2011-2015: Eurozone crisis, oil price swings, taper tantrum.",
    "Focus on a specific policy or shock (e.g. oil crisis, Fed rate cycle) rather than the same events every time.",
]

# NBER Macrohistory: historical eras (1860s–1940s)
NBER_ERA_SUGGESTIONS = [
    "Focus on 1929-1933: Great Depression, stock market crash.",
    "Focus on 1893-1897: Panic of 1893 and aftermath.",
    "Focus on 1907-1908: Panic of 1907.",
    "Focus on 1914-1918: World War I.",
    "Focus on 1920-1921: Post-WWI recession.",
]


def build_puzzle_seed_prompt(
    *,
    requested_source: str,
    series_list: str,
    examples_str: str,
    avoid_2019_2021_covid: bool = False,
) -> str:
    """
    Build the prompt for generating a single puzzle seed (FRED or Google Trends).
    Call from seed_generator when calling the LLM. Designed for variety and robustness.
    """
    era_hint = (
        random.choice(NBER_ERA_SUGGESTIONS)
        if requested_source == "nber"
        else random.choice(ERA_SUGGESTIONS)
    )

    if requested_source == "fred":
        source_instruction = f'''You MUST output a FRED seed with "source": "fred".
- seriesId: from this list ONLY: {series_list}
- startDate, endDate: YYYY-MM-DD (must match the event's timeline)
- correctEvent, acceptableAnswers (list), explanation, hints (array of 4 strings, increasingly obvious)

Variety rules:
- Pick a UNIQUE combination: do not repeat the same series + date range as the examples.
- Prefer different events (e.g. 1980s recession, dot-com bust, 1990s recession, 2008 crisis) and vary the decade.
- Get data dating back very far into the 1800s and 1700s
- This time: {era_hint}'''
    elif requested_source == "nber":
        source_instruction = f'''You MUST output an NBER Macrohistory seed with "source": "nber".
- seriesId: from this list ONLY (format chapter/filename, e.g. 01/a01005a): {series_list}
- startDate, endDate: YYYY-MM-DD (NBER data is mostly 1860s–1940s; pick a range within the series coverage)
- correctEvent, acceptableAnswers (list), explanation, hints (array of 4 strings, increasingly obvious)

NBER data is historical (1800s–1940s). Pick a well-known causal event in that era, e.g.:
Panic of 1893, Panic of 1907, World War I, post-WWI recession, Great Depression (1929–1933), New Deal era, Dust Bowl.
- This time: {era_hint}'''
    else:
        source_instruction = f'''You MUST output a Google Trends seed with "source": "google_trends".
- searchTerm: a phrase people actually search (e.g. "bankruptcy", "foreclosure", "gold price", "oil crisis", "recession", "layoffs")
- startDate, endDate: YYYY-MM-DD
- correctEvent, acceptableAnswers (list), explanation, hints (array of 4 strings, increasingly obvious)

Variety rules:
- Pick a UNIQUE search term and event: do not repeat toilet paper, face mask, or COVID. Vary the decade and type of event.
- This time: {era_hint}'''

    avoid_instruction = ""
    if avoid_2019_2021_covid:
        avoid_instruction = """

Session constraint: This session has already had puzzles from 2019-2021 or COVID-19. You MUST NOT use date range 2019-2021. Use a different range (e.g. 2007-2009, 2001-2003, 1980-1983). You MUST NOT use COVID-19 pandemic as correctEvent; use e.g. 2008 financial crisis, dot-com bust, early 1980s recession, oil crisis."""

    return f"""You are creating a single "causal guessr" puzzle: a time-series chart where the player must guess what real-world event caused the trend. They get 4 guesses and a hint after each wrong guess. Do not give away the answer until the 4th hint.

{source_instruction}{avoid_instruction}

Output exactly one JSON object with the required keys. No other text, no markdown, no code block.
Examples (format only; do not copy content—pick different series/terms, dates, and events):
{examples_str}

Reply with only the JSON object."""


def build_guess_evaluation_prompt(
    guess: str,
    correct_event: str,
    other_acceptable_str: str,
) -> str:
    """
    Build the prompt for evaluating whether a user's guess is semantically correct.
    Call from guess_evaluator when the guess is not in the acceptable list.
    """
    return f"""The correct answer for this puzzle is: "{correct_event}".
Other acceptable answers include: {other_acceptable_str}.

The user guessed: "{guess}"

Is the user's guess correct? (Same event, same meaning, or an equivalent way to say the correct answer.)
Reply with ONLY the word true or false, nothing else."""
