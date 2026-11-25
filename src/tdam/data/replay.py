"""Live-style replay: consume saved telemetry in ARRIVAL order and score at decision ticks.

Semantics
  * Ingestion is strictly by arrival time. At tick t every record with arrival <= t has been seen.
  * Late records (event bin still inside the lookback) are folded into the open bins and are visible
    to later ticks only; an alert already issued is never moved or rewritten.
  * Stale records (event bin already older than the window) are dropped and counted.
  * Duplicates: an event_id is remembered only while its bin is inside the window (bounded state);
    a repeat delivery is counted and ignored.
  * Idle entities: state older than the window is pruned, so memory is O(entities x window). An
    entity with no live state is scored with an empty window (true zeros while it is monitored).
  * Cooldown grouping keeps the first alert's timestamp.
Replay features are identical to the batch features at the same decision time (tested).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from tdam.data.schema import Event
from tdam.evaluation.latency import Timer
from tdam.temporal.binning import add_gap_channel
from tdam.timeutil import Timeline, ceil_div


class Replayer:
    def __init__(self, entities: list[str], timeline: Timeline, lookback: int, cover: np.ndarray | None = None):
        self.entities, self.tl, self.w = entities, timeline, lookback
        self.index = {e: i for i, e in enumerate(entities)}
        self.cover = cover
        # entity -> bin -> [n, nf, set(dst)]
        self.state: dict[int, dict[int, list]] = defaultdict(dict)
        self.seen: dict[int, dict[int, set]] = defaultdict(lambda: defaultdict(set))
        self.stats = {"ingested": 0, "duplicates": 0, "stale_dropped": 0, "unknown_entity": 0}
        self.clock_bin = -(10**9)

    def ingest(self, ev: Event, now_bin: int) -> None:
        e = self.index.get(ev.entity)
        if e is None:
            self.stats["unknown_entity"] += 1
            return
        b = int(ceil_div(ev.event_time - self.tl.t0, self.tl.bin_seconds))
        if b <= now_bin - self.w:
            self.stats["stale_dropped"] += 1
            return
        ids = self.seen[e][b]
        if ev.event_id in ids:
            self.stats["duplicates"] += 1
            return
        ids.add(ev.event_id)
        slot = self.state[e].setdefault(b, [0.0, 0.0, set()])
        slot[0] += 1
        slot[1] += 1.0 if ev.outcome == "fail" else 0.0
        slot[2].add(ev.dst)
        self.stats["ingested"] += 1

    def prune(self, t: int) -> None:
        for e in list(self.state):
            for b in [b for b in self.state[e] if b <= t - self.w]:
                del self.state[e][b]
                self.seen[e].pop(b, None)
            if not self.state[e]:
                del self.state[e]
                self.seen.pop(e, None)

    def snapshot(self, t: int, entities_idx: list[int]):
        """Window arrays [N, W, 6] for the given entities at decision bin t."""
        w = self.w
        raw = np.zeros((len(entities_idx), w, 3), dtype=np.float32)
        obs = np.zeros((len(entities_idx), w), dtype=bool)
        lengths = np.full(len(entities_idx), min(w, t + 1), dtype=np.int64)
        for r, e in enumerate(entities_idx):
            for b, (n, nf, dsts) in self.state.get(e, {}).items():
                k = b - (t - w + 1)
                if 0 <= k < w and b <= t:
                    raw[r, k] = (n, nf, len(dsts))
            starts = t - w + 1 + np.arange(w)
            valid = starts >= 0
            if self.cover is not None:
                obs[r, valid] = self.cover[e, starts[valid]]
            else:
                obs[r, valid] = True
        n = raw[..., 0]
        gap = add_gap_channel(n)
        X = np.stack([n, raw[..., 1], n - raw[..., 1], raw[..., 2], gap, obs.astype(np.float32)], axis=-1)
        pad = np.arange(w)[None, :] < (w - lengths[:, None])
        X[pad] = 0.0
        return X.astype(np.float32), lengths


def replay(
    events: list[Event],
    entities: list[str],
    timeline: Timeline,
    cover: np.ndarray,
    bundle,
    ticks: np.ndarray,
    timer: Timer | None = None,
):
    """Replay events by arrival and score every monitored entity at each tick bin.

    Returns a list of prediction rows (dicts) and replay statistics.
    """
    rep = Replayer(entities, timeline, bundle.lookback, cover)
    ordered = sorted(events, key=lambda e: (e.arrival_time, e.event_id))
    ptr, preds = 0, []
    timer = timer or Timer()
    for t in ticks:
        t = int(t)
        t_s = timeline.time_of(t)
        with timer.measure("ingest_ms"):
            while ptr < len(ordered) and ordered[ptr].arrival_time <= t_s:
                rep.ingest(ordered[ptr], t)
                ptr += 1
            rep.prune(t)
        live = [e for e in range(len(entities)) if cover[e, t]]
        if not live:
            continue
        with timer.measure("feature_ms"):
            X, lengths = rep.snapshot(t, live)
        with timer.measure("score_ms"):
            scores = bundle.predict(X, lengths)
        for r, e in enumerate(live):
            row = {"entity": entities[e], "t": t, "time_s": t_s}
            row.update({m: float(s[r]) for m, s in scores.items()})
            preds.append(row)
    return preds, rep.stats
