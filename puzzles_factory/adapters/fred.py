"""
FRED adapter: fetch observations via cached FRED client and build puzzle struct.
Adds viz hints (chartType, yLabel) from viz_hints when not in metadata.
"""

from puzzles_factory.base import BasePuzzleAdapter
from puzzles_factory.viz_hints import get_viz_hints


class FredAdapter(BasePuzzleAdapter):
    """Build puzzles from FRED series (observations fetched via cache)."""

    @property
    def source_id(self) -> str:
        return "fred"

    def fetch_observations(self, data: dict) -> list[dict]:
        series_id = data.get("seriesId")
        start = data.get("startDate")
        end = data.get("endDate")
        if not series_id or not start or not end:
            raise ValueError("FRED data must include seriesId, startDate, endDate")

        from api.fred_client import get_observations_cached

        return get_observations_cached(series_id, start, end)

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
