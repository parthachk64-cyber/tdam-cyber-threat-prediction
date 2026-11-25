"""Events -> bin rows -> point-in-time windows.

A *bin row* is (entity, b, avail, n, nf, nd): n events and nf failures that
happened in bin b and become visible to the defender at decision bin `avail`
(avail >= b); nd counts destinations seen for the first time in bin b.

Eligibility rule for decision time t, entity e, lookback W:
    event bin b in (t - W, t]   AND   arrival no later than t  (avail <= t)
so a record stamped 10:16 but ingested 10:18 is invisible to the 10:17 decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tdam.data.schema import Event
from tdam.timeutil import Timeline, ceil_div

RAW_CH = 3  # n, nf, nd
CHANNELS = ("events", "fail", "success", "uniq_dst", "gap_min", "observed")


@dataclass
class BinRows:
    entity: np.ndarray  # int index into the entity list
    b: np.ndarray
    avail: np.ndarray
    n: np.ndarray
    nf: np.ndarray
    nd: np.ndarray

    def __len__(self) -> int:
        return len(self.b)

    def to_frame(self, entities: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "entity": [entities[i] for i in self.entity],
                "b": self.b,
                "avail": self.avail,
                "n": self.n,
                "nf": self.nf,
                "nd": self.nd,
            }
        )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, entities: list[str]) -> "BinRows":
        index = {e: i for i, e in enumerate(entities)}
        keep = frame["entity"].isin(index)
        frame = frame[keep]
        return cls(
            entity=frame["entity"].map(index).to_numpy(dtype=np.int64),
            b=frame["b"].to_numpy(dtype=np.int64),
            avail=frame["avail"].to_numpy(dtype=np.int64),
            n=frame["n"].to_numpy(dtype=np.float64),
            nf=frame["nf"].to_numpy(dtype=np.float64),
            nd=frame["nd"].to_numpy(dtype=np.float64),
        )

    def upto_avail(self, t_max: int) -> "BinRows":
        """Rows visible by decision bin t_max (used to prove no look-ahead)."""
        m = self.avail <= t_max
        return BinRows(*(getattr(self, f)[m] for f in ("entity", "b", "avail", "n", "nf", "nd")))


def events_to_rows(events: list[Event], entities: list[str], timeline: Timeline):
    """Deduplicate by event_id (earliest delivery wins) and aggregate to bin rows.

    Returns (BinRows, stats). Events outside the timeline or for unknown entities are dropped
    and counted so a validator can report them.
    """
    stats = {"input": len(events), "duplicates": 0, "unknown_entity": 0, "outside_timeline": 0}
    if not events:
        empty = np.zeros(0, dtype=np.int64)
        z = np.zeros(0)
        return BinRows(empty, empty, empty, z, z, z), stats
    frame = pd.DataFrame(
        {
            "id": [e.event_id for e in events],
            "entity": [e.entity for e in events],
            "et": [e.event_time for e in events],
            "at": [e.arrival_time for e in events],
            "dst": [e.dst for e in events],
            "fail": [1.0 if e.outcome == "fail" else 0.0 for e in events],
        }
    )
    frame = frame.sort_values(["at", "id"], kind="stable")
    before = len(frame)
    frame = frame.drop_duplicates("id", keep="first")
    stats["duplicates"] = before - len(frame)
    index = {e: i for i, e in enumerate(entities)}
    frame["e"] = frame["entity"].map(index)
    stats["unknown_entity"] = int(frame["e"].isna().sum())
    frame = frame.dropna(subset=["e"])
    frame["b"] = ceil_div(frame["et"].to_numpy() - timeline.t0, timeline.bin_seconds)
    frame["a"] = np.maximum(
        ceil_div(frame["at"].to_numpy() - timeline.t0, timeline.bin_seconds), frame["b"].to_numpy()
    )
    inside = (frame["b"] >= 0) & (frame["b"] < timeline.n_bins)
    stats["outside_timeline"] = int((~inside).sum())
    frame = frame[inside]
    frame["e"] = frame["e"].astype(np.int64)
    counts = (
        frame.groupby(["e", "b", "a"], sort=False)
        .agg(n=("fail", "size"), nf=("fail", "sum"))
        .reset_index()
    )
    counts["nd"] = 0.0
    firsts = frame.groupby(["e", "b", "dst"], sort=False)["a"].min().reset_index()
    nd_rows = firsts.groupby(["e", "b", "a"], sort=False).size().reset_index(name="nd")
    nd_rows["n"] = 0.0
    nd_rows["nf"] = 0.0
    merged = (
        pd.concat([counts, nd_rows[["e", "b", "a", "n", "nf", "nd"]]])
        .groupby(["e", "b", "a"], sort=True)
        .sum()
        .reset_index()
    )
    rows = BinRows(
        entity=merged["e"].to_numpy(np.int64),
        b=merged["b"].to_numpy(np.int64),
        avail=merged["a"].to_numpy(np.int64),
        n=merged["n"].to_numpy(np.float64),
        nf=merged["nf"].to_numpy(np.float64),
        nd=merged["nd"].to_numpy(np.float64),
    )
    return rows, stats


def coverage_mask(coverage: dict, entities: list[str], timeline: Timeline) -> np.ndarray:
    """[E, n_bins] bool. True = the source was monitored in that bin."""
    from tdam.timeutil import parse_iso

    mask = np.zeros((len(entities), timeline.n_bins), dtype=bool)
    for i, e in enumerate(entities):
        for lo, hi in coverage["entities"].get(e, []):
            a = int(timeline.bin_of(parse_iso(lo)))
            b = int(timeline.bin_of(parse_iso(hi)))
            mask[i, max(a, 0) : min(b, timeline.n_bins)] = True
    return mask


def add_gap_channel(raw_n: np.ndarray) -> np.ndarray:
    """Minutes since the last non-empty bin inside the window, per bin: [N, W]."""
    w = raw_n.shape[1]
    pos = np.arange(w)[None, :]
    last = np.maximum.accumulate(np.where(raw_n > 0, pos, -1), axis=1)
    return np.where(last >= 0, pos - last, pos + 1).astype(np.float32)


def finalize_channels(raw: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """raw [N, W, 3] (n, nf, nd) + observed [N, W] -> X [N, W, 6]."""
    n, nf, nd = raw[..., 0], raw[..., 1], raw[..., 2]
    gap = add_gap_channel(n)
    out = np.stack([n, nf, n - nf, nd, gap, observed.astype(np.float32)], axis=-1)
    return out.astype(np.float32)


def window_raw_for_entity(rows: BinRows, ent: int, n_bins: int, w: int) -> np.ndarray:
    """Dense [n_bins, W, 3] windows for one entity at every decision bin.

    Position k of the window ending at t holds bin t - W + 1 + k. A row contributes to
    decisions t in [avail, b + W - 1].
    """
    sel = rows.entity == ent
    b, a = rows.b[sel], rows.avail[sel]
    vals = np.stack([rows.n[sel], rows.nf[sel], rows.nd[sel]], axis=1)
    out = np.zeros((n_bins, w, RAW_CH), dtype=np.float64)
    if not len(b):
        return out
    flat = np.zeros(n_bins * w * RAW_CH)
    for j in range(w):  # j = t - b
        t = b + j
        m = (a <= t) & (t < n_bins)
        if not m.any():
            continue
        k = w - 1 - j
        base = (t[m] * w + k) * RAW_CH
        for c in range(RAW_CH):
            flat += np.bincount(base + c, weights=vals[m, c], minlength=flat.size)
    return flat.reshape(n_bins, w, RAW_CH)


def build_entity_windows(
    rows: BinRows, ent: int, timeline: Timeline, w: int, cover: np.ndarray, decisions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X [N, W, 6], lengths [N]) for the given decision bins of one entity.

    Bins before bin 0 are padding (not unknown coverage): `lengths` records valid bins and the
    padded positions are zero in every channel including `observed`.
    """
    raw = window_raw_for_entity(rows, ent, timeline.n_bins, w)[decisions]
    starts = decisions[:, None] - w + 1 + np.arange(w)[None, :]
    valid = starts >= 0
    obs = np.zeros(starts.shape, dtype=bool)
    obs[valid] = cover[starts[valid]]
    raw = raw * valid[..., None]
    X = finalize_channels(raw, obs)
    X[~valid] = 0.0
    lengths = valid.sum(axis=1).astype(np.int64)
    return X, lengths
