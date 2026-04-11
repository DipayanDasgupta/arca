"""arca.utils — Shared utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def save_json(data: dict, path: str | Path) -> None:
    """Save a dict as pretty-printed JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def smooth(data: list[float], window: int = 5) -> list[float]:
    """Simple moving-average smoothing."""
    import numpy as np
    if len(data) < window:
        return data
    kernel = [1.0 / window] * window
    result = []
    for i in range(len(data) - window + 1):
        result.append(sum(data[i:i+window]) / window)
    return result


def timestamp() -> str:
    """Return a sortable timestamp string like 20260412_153012."""
    return time.strftime("%Y%m%d_%H%M%S")


class Timer:
    """Simple context-manager timer."""
    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start

    def __str__(self):
        return f"{self.elapsed:.2f}s"


__all__ = ["save_json", "load_json", "smooth", "timestamp", "Timer"]