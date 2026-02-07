"""
Default visualization hints per source.
Adapters merge these into the built puzzle when metadata does not override.
"""


def _default_viz_hints(source: str, metadata: dict) -> dict:
    """
    Return default chartType, yLabel, and optional yLimits for a source.
    Internal: used by adapters when building the puzzle.
    """
    if source == "fred":
        return {
            "chartType": "line",
            "yLabel": metadata.get("title") or "Value",
        }
    if source == "google_trends":
        return {
            "chartType": "line",
            "yLabel": "Search interest (0–100)",
            "yLimits": (0, 100),
        }
    return {
        "chartType": "line",
        "yLabel": metadata.get("title") or "Value",
    }


def get_viz_hints(source: str, metadata: dict) -> dict:
    """
    Return viz hints for the puzzle: metadata overrides (chartType, yLabel, yLimits) take
    precedence over source defaults.
    """
    hints = _default_viz_hints(source, metadata)
    if metadata.get("chartType") is not None:
        hints["chartType"] = metadata["chartType"]
    if metadata.get("yLabel") is not None:
        hints["yLabel"] = metadata["yLabel"]
    if metadata.get("yLimits") is not None:
        hints["yLimits"] = metadata["yLimits"]
    return hints
