"""
NBER Macrohistory adapter: fetch observations via cached NBER client and build puzzle struct.
"""

from puzzles_factory.base import BasePuzzleAdapter
from puzzles_factory.viz_hints import get_viz_hints


class NberAdapter(BasePuzzleAdapter):
    """Build puzzles from NBER Macrohistory .db series."""

    @property
    def source_id(self) -> str:
        return "nber"

    def fetch_observations(self, data: dict) -> list[dict]:
        series_id = data.get("seriesId")
        start = data.get("startDate")
        end = data.get("endDate")
        if not series_id or not start or not end:
            raise ValueError("NBER data must include seriesId, startDate, endDate")

        from api.nber import get_observations_cached

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
