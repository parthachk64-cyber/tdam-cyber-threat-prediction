"""Convex blend of the LSTM and tree scores.

score = w * p_lstm + (1 - w) * p_tree, 0 <= w <= 1. The weight is learned ONLY from chronological
out-of-fold predictions of the development data (never from the calibration or test partitions).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

GRID = np.round(np.linspace(0.0, 1.0, 11), 2)


def blend(p_lstm: np.ndarray, p_tree: np.ndarray, w: float) -> np.ndarray:
    if not 0.0 <= w <= 1.0:
        raise ValueError("blend weight must lie in [0, 1]")
    return w * p_lstm + (1.0 - w) * p_tree


def learn_weight(p_lstm: np.ndarray, p_tree: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """Grid search on out-of-fold average precision. Single-class OOF -> equal weights."""
    if len(y) == 0 or y.min() == y.max():
        return 0.5, {"reason": "single-class OOF slice; equal weights", "grid": {}}
    grid = {float(w): float(average_precision_score(y, blend(p_lstm, p_tree, w))) for w in GRID}
    best = max(grid.values())
    ties = [w for w, v in grid.items() if abs(v - best) < 1e-12]
    w = min(ties, key=lambda x: abs(x - 0.5))
    return w, {"reason": "max OOF average precision", "grid": grid}
