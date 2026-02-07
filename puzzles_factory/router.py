"""
Route puzzle build requests to the correct source adapter and return puzzle JSON.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from puzzles_factory.base import BasePuzzleAdapter

# Registry: source_id -> adapter instance
_registry: dict[str, "BasePuzzleAdapter"] = {}


def register_adapter(adapter: "BasePuzzleAdapter") -> None:
    """Register an adapter for its source_id. Re-registering overwrites."""
    _registry[adapter.source_id] = adapter


def _get_adapter(source: str) -> "BasePuzzleAdapter":
    """Return adapter for source; raise if unknown."""
    adapter = _registry.get(source)
    if adapter is None:
        raise ValueError(f"Unknown puzzle source: {source}. Registered: {list(_registry)}")
    return adapter


def build_puzzle(metadata: dict) -> dict:
    """
    Build a full puzzle struct from puzzle metadata (e.g. one entry from puzzles.json).

    Dispatches to the adapter for metadata["source"]; the adapter fetches observations
    (using cache when applicable) and merges them into the canonical puzzle shape.

    Args:
        metadata: Must contain "source" and "data" (source-specific). Typically also
                  id, title, correctEvent, acceptableAnswers, explanation.

    Returns:
        Puzzle dict with metadata fields and "series" (list of {date, value}) for plotting.
    """
    source = metadata.get("source")
    if not source:
        raise ValueError("Puzzle metadata must include 'source'")
    data = metadata.get("data")
    if not data:
        raise ValueError("Puzzle metadata must include 'data'")

    adapter = _get_adapter(source)
    observations = adapter.fetch_observations(data)
    return adapter.build_puzzle(metadata, observations)
