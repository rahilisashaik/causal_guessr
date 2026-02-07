"""
Abstract base for source-specific puzzle adapters.
Each adapter knows how to fetch observations and build the canonical puzzle struct.
"""

import math
from abc import ABC, abstractmethod


class BasePuzzleAdapter(ABC):
    """Adapter for one API source: fetch observations and build puzzle JSON."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Source identifier (e.g. 'fred', 'google_trends')."""
        ...

    @abstractmethod
    def fetch_observations(self, data: dict) -> list[dict]:
        """
        Fetch time-series observations for this source using source-specific data.

        Args:
            data: Source-specific payload (e.g. {"seriesId", "startDate", "endDate"} for FRED).

        Returns:
            List of {"date": "YYYY-MM-DD", "value": "..."}.
        """
        ...

    @abstractmethod
    def build_puzzle(self, metadata: dict, observations: list[dict]) -> dict:
        """
        Build the canonical puzzle JSON struct from metadata and observations.

        Args:
            metadata: Puzzle definition (id, source, title, correctEvent, acceptableAnswers, explanation, data).
            observations: List of {"date", "value"} from fetch_observations.

        Returns:
            Full puzzle dict: metadata fields + optional "series" (normalized observations).
        """
        ...

    @staticmethod
    def _normalize_series(observations: list[dict]) -> list[dict]:
        """
        Normalize observations: parse numeric values; use NaN for missing (".", "NA")
        so the full date range is preserved (e.g. NBER with gaps).
        """
        out = []
        for ob in observations:
            val = ob.get("value", ".")
            if val == "." or val is None or (isinstance(val, str) and val.strip().upper() == "NA"):
                out.append({"date": ob["date"], "value": math.nan})
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                num = math.nan
            out.append({"date": ob["date"], "value": num})
        return out
