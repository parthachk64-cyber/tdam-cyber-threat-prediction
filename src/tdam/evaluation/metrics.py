"""Window-level metrics with explicit handling of single-class slices."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _div(a: float, b: float):
    return None if b == 0 else a / b


def window_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    """Confusion counts and rates at `thr`. Undefined quantities are None, never silently 0 or 1."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n = len(y)
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    single = n == 0 or y.min() == y.max()
    return {
        "n": n,
        "positives": int(y.sum()),
        "prevalence": _div(int(y.sum()), n),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": _div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": None if single else float(average_precision_score(y, score)),
        "roc_auc": None if single else float(roc_auc_score(y, score)),
        "always_negative_accuracy": _div(int((y == 0).sum()), n),
    }


def f1_optimal_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising F1 (calibration data only). Falls back to 'never alert' without positives."""
    y = np.asarray(y).astype(int)
    if len(y) == 0 or y.sum() == 0 or y.min() == y.max():
        return float(np.nextafter(np.max(score), np.inf)) if len(score) else float("inf")
    p, r, t = precision_recall_curve(y, score)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f))])


def lead_time_summary(leads_min: list[float]) -> dict:
    if not leads_min:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    a = np.asarray(leads_min, dtype=float)
    return {"n": len(a), "median": float(np.median(a)), "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75))}
