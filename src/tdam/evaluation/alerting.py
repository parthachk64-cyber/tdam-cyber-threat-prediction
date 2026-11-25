"""Scores -> grouped alerts -> matched ledger; threshold selection on calibration data only."""

from __future__ import annotations

import numpy as np

from tdam.config import Config
from tdam.data.schema import Incident
from tdam.evaluation.matching import MatchResult, match_alerts
from tdam.evaluation.metrics import f1_optimal_threshold
from tdam.temporal.labels import OK, RECOVERY
from tdam.temporal.splits import Partition, partition_mask
from tdam.temporal.windows import WindowSet


def issue_alerts(ent: np.ndarray, t_s: np.ndarray, score: np.ndarray, thr: float, cooldown_s: int) -> np.ndarray:
    """Indices of issued alerts. After an alert on an entity, further threshold crossings inside the
    cooldown are grouped into it. An issued alert keeps the decision time at which it fired."""
    cross = np.nonzero(score >= thr)[0]
    if not len(cross):
        return cross
    order = cross[np.lexsort((t_s[cross], ent[cross]))]
    kept = []
    last_e, last_t = -1, -(10**18)
    for i in order:
        e, t = int(ent[i]), int(t_s[i])
        if e != last_e or t - last_t >= cooldown_s:
            kept.append(i)
            last_e, last_t = e, t
    return np.asarray(kept, dtype=np.int64)


def evaluate_alerts(
    ws: WindowSet,
    scores: np.ndarray,
    thr: float,
    part: Partition,
    incidents: list[Incident],
    entities: list[str],
    cfg: Config,
) -> dict:
    """Event-level evaluation of one operating point inside one partition."""
    w = cfg.protocol.lookback_bins
    h_bins = cfg.protocol.horizon_minutes * 60 // cfg.protocol.bin_seconds
    in_part = partition_mask(ws.t, part, w, h_bins)
    sub = ws.subset(in_part)
    sc = scores[in_part]
    idx = issue_alerts(sub.ent, sub.t_s, sc, thr, cfg.protocol.cooldown_minutes * 60)
    idx = idx[sub.status[idx] != RECOVERY]
    alerts = [(entities[int(sub.ent[i])], int(sub.t_s[i])) for i in idx]
    result: MatchResult = match_alerts(alerts, incidents, cfg.horizon_s)
    decision_hours = cfg.protocol.decision_stride * cfg.protocol.bin_seconds / 3600.0
    exposure = float((sub.status == OK).sum()) * decision_hours
    counts = result.counts()
    counts["exposure_host_hours"] = exposure
    counts["unmatched_per_host_hour"] = counts["unmatched_alerts"] / exposure if exposure > 0 else None
    counts["threshold"] = float(thr)
    return {"counts": counts, "result": result, "alerts": alerts}


def select_budget_threshold(
    ws: WindowSet,
    scores: np.ndarray,
    part: Partition,
    incidents: list[Incident],
    entities: list[str],
    cfg: Config,
) -> float:
    """Lowest threshold whose unmatched-warning rate per host-hour stays within the alert budget.

    If even the highest score exceeds the budget (or the score is constant), no alerts are issued:
    the threshold is set above the maximum score."""
    budget = cfg.alerting.budget_per_host_hour
    w, h = cfg.protocol.lookback_bins, cfg.protocol.horizon_minutes * 60 // cfg.protocol.bin_seconds
    in_part = partition_mask(ws.t, part, w, h)
    sc = scores[in_part]
    if not len(sc):
        return float("inf")
    cands = np.unique(np.quantile(sc, np.concatenate([np.linspace(0.0, 0.99, 100), np.linspace(0.99, 1.0, 101)])))
    top = float(np.nextafter(sc.max(), np.inf))
    cands = np.append(cands, top)

    def rate(thr):
        return evaluate_alerts(ws, scores, thr, part, incidents, entities, cfg)["counts"]["unmatched_per_host_hour"] or 0.0

    lo, hi = 0, len(cands) - 1  # cands[hi] issues no alerts, hence always within budget
    while lo < hi:
        mid = (lo + hi) // 2
        if rate(cands[mid]) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return float(cands[lo])


def select_f1_threshold(ws: WindowSet, scores: np.ndarray, part: Partition, cfg: Config) -> float:
    w, h = cfg.protocol.lookback_bins, cfg.protocol.horizon_minutes * 60 // cfg.protocol.bin_seconds
    m = partition_mask(ws.t, part, w, h) & ws.usable
    return f1_optimal_threshold(ws.y[m], scores[m])
