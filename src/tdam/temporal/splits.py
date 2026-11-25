"""Chronological, purged partitions.

Partition p covers bins [start, end). A decision t belongs to p only when its whole footprint
[t - W + 1, t + H] lies inside p, i.e. t >= start + W - 1 and t + H <= end - 1. That purges
windows whose lookback or target crosses a boundary, so no window (and no incident's positive
windows) can appear in two partitions. History is NOT carried across partition boundaries in
the fixed-holdout evaluation; the online replay carries state because it is causal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdam.data.schema import Incident
from tdam.timeutil import Timeline

ORDER = ("train", "val", "calib", "test")


@dataclass(frozen=True)
class Partition:
    name: str
    start: int
    end: int

    def first_t(self, w: int) -> int:
        return self.start + w - 1

    def last_t(self, h: int) -> int:
        return self.end - 1 - h


def make_partitions(n_bins: int, fractions: dict[str, float]) -> dict[str, Partition]:
    edges = [0]
    acc = 0.0
    for name in ORDER[:-1]:
        acc += fractions[name]
        edges.append(int(round(acc * n_bins)))
    edges.append(n_bins)
    return {name: Partition(name, edges[i], edges[i + 1]) for i, name in enumerate(ORDER)}


def span_mask(t: np.ndarray, start: int, end: int, w: int, h: int) -> np.ndarray:
    return (t >= start + w - 1) & (t + h <= end - 1)


def partition_mask(t: np.ndarray, part: Partition, w: int, h: int) -> np.ndarray:
    return span_mask(t, part.start, part.end, w, h)


def assign_incidents(
    incidents: list[Incident], parts: dict[str, Partition], timeline: Timeline, w: int, h_min: int
) -> dict[str, list[Incident]]:
    """An incident belongs to the partition whose decision span contains its onset (t, t+H]."""
    h_bins = h_min * 60 // timeline.bin_seconds
    out: dict[str, list[Incident]] = {p: [] for p in parts}
    out["purged"] = []
    for inc in incidents:
        placed = False
        for name, p in parts.items():
            lo = timeline.time_of(p.first_t(w))
            hi = timeline.time_of(p.last_t(h_bins) + h_bins)
            if lo < inc.onset <= hi:
                out[name].append(inc)
                placed = True
                break
        if not placed:
            out["purged"].append(inc)
    return out


def manifest(parts, assigned, timeline: Timeline, w: int, h_bins: int) -> dict:
    from tdam.timeutil import iso

    return {
        "rule": "decision t in partition iff t >= start + W - 1 and t + H <= end - 1",
        "lookback_bins": w,
        "horizon_bins": h_bins,
        "partitions": {
            n: {
                "start": iso(timeline.time_of(p.start)),
                "end": iso(timeline.time_of(p.end)),
                "first_decision": iso(timeline.time_of(p.first_t(w))),
                "last_decision": iso(timeline.time_of(p.last_t(h_bins))),
                "incident_ids": sorted(i.incident_id for i in assigned[n]),
            }
            for n, p in parts.items()
        },
        "purged_incident_ids": sorted(i.incident_id for i in assigned["purged"]),
    }
