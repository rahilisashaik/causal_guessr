"""
Plot a puzzle's time series with matplotlib.
Uses chartType, yLabel, yLimits, and series from the built puzzle (no source-specific logic).
"""

import io
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend for server (no display)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def _parse_dates(series: list[dict]) -> np.ndarray:
    """Parse date strings to matplotlib-friendly format."""
    from datetime import datetime

    dates = [datetime.strptime(ob["date"], "%Y-%m-%d") for ob in series]
    return np.array(dates)


def _get_values(series: list[dict]) -> np.ndarray:
    """Extract values as float array."""
    return np.array([float(ob["value"]) for ob in series], dtype=float)


def _draw_line(ax: Any, dates: np.ndarray, values: np.ndarray) -> None:
    """Draw a line chart."""
    ax.plot(dates, values, marker="o", markersize=4, linestyle="-")


def _draw_area(ax: Any, dates: np.ndarray, values: np.ndarray) -> None:
    """Draw an area chart."""
    ax.fill_between(dates, values, alpha=0.4)
    ax.plot(dates, values, linewidth=1.5)


def _draw_bar(ax: Any, dates: np.ndarray, values: np.ndarray) -> None:
    """Draw a bar chart. Width in days, scaled by point spacing."""
    delta = (dates.max() - dates.min()).days
    width_days = (delta / max(len(dates), 1)) * 0.8 if delta else 1.0
    ax.bar(dates, values, width=width_days)


def plot(
    puzzle: dict,
    path: str | Path | None = None,
    *,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
) -> None:
    """
    Plot the puzzle's series using chartType, yLabel, and optional yLimits.

    Args:
        puzzle: Built puzzle dict with "series", "chartType", "yLabel", and optional "yLimits".
        path: If set, save figure to this path; otherwise show interactively.
        figsize: Figure size (width, height).
        title: Chart title. If None, uses puzzle["title"] (no causal spoilers).
    """
    series = puzzle.get("series")
    if not series:
        raise ValueError("Puzzle has no 'series' to plot")

    chart_type = puzzle.get("chartType", "line")
    y_label = puzzle.get("yLabel", "Value")
    y_limits = puzzle.get("yLimits")  # optional (ymin, ymax)
    x_label = "Date"

    dates = _parse_dates(series)
    values = _get_values(series)

    fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 12)))
    plt.xticks(rotation=45)

    if chart_type == "line":
        _draw_line(ax, dates, values)
    elif chart_type == "area":
        _draw_area(ax, dates, values)
    elif chart_type == "bar":
        _draw_bar(ax, dates, values)
    else:
        _draw_line(ax, dates, values)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_title(title if title is not None else puzzle.get("title", ""))

    fig.tight_layout()

    if path is not None:
        if isinstance(path, io.BytesIO):
            fig.savefig(path, format="png", dpi=150)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_to_bytes(
    puzzle: dict,
    *,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
    dpi: int = 150,
) -> bytes:
    """Render the puzzle chart to PNG bytes (e.g. for serving in a web UI)."""
    buf = io.BytesIO()
    plot(puzzle, path=buf, figsize=figsize, title=title)
    buf.seek(0)
    return buf.read()
