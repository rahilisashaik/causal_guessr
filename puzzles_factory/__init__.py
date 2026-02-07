"""
Build canonical puzzle JSON structs from multiple API sources (FRED, Google Trends, etc.).
Route by source and delegate to source-specific adapters.
"""

from puzzles_factory.base import BasePuzzleAdapter
from puzzles_factory.router import build_puzzle, register_adapter

# Register built-in adapters so build_puzzle(metadata) works for known sources
from puzzles_factory.adapters.fred import FredAdapter

register_adapter(FredAdapter())

__all__ = [
    "BasePuzzleAdapter",
    "build_puzzle",
    "register_adapter",
]
