"""
Google Trends adapter: fetch search interest via pytrends and build puzzle struct.
Uses viz hints (Search interest 0–100, yLimits) from viz_hints.
"""

from puzzles_factory.base import BasePuzzleAdapter
from puzzles_factory.viz_hints import get_viz_hints


class GoogleTrendsAdapter(BasePuzzleAdapter):
    """Build puzzles from Google Trends (interest over time for a search term)."""

    @property
    def source_id(self) -> str:
        return "google_trends"

    def fetch_observations(self, data: dict) -> list[dict]:
        keyword = data.get("searchTerm")
        start = data.get("startDate")
        end = data.get("endDate")
        if not keyword or not start or not end:
            raise ValueError("Google Trends data must include searchTerm, startDate, endDate")

        from api.google_trends_client import get_interest_over_time_cached

        geo = data.get("geo") or ""
        return get_interest_over_time_cached(keyword, start, end, geo)

    def build_puzzle(self, metadata: dict, observations: list[dict]) -> dict:
        series = self._normalize_series(observations)
        viz = get_viz_hints(self.source_id, metadata)
        return {
            "id": metadata["id"],
            "source": metadata["source"],
            "title": metadata["title"],
            "correctEvent": metadata["correctEvent"],
            "acceptableAnswers": metadata["acceptableAnswers"],
            "explanation": metadata["explanation"],
            "data": metadata["data"],
            "series": series,
            "chartType": viz["chartType"],
            "yLabel": viz["yLabel"],
            **({"yLimits": viz["yLimits"]} if viz.get("yLimits") is not None else {}),
        }
