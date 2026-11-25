"""Forecast labels.

Decision time t (seconds), entity e, horizon H, recovery R:
  y = 1  iff an incident onset of e lies in (t, t + H]
  status ACTIVE   an incident of e is in progress at t         (excluded from training/metrics)
  status RECOVERY an incident of e ended within the last R      (excluded)
  status CENSORED the future window (t, t+H] is not fully inside the observed stream (excluded)
  status OK       otherwise
`known_at` is the earliest time the label of the window is settled for a defender: the horizon has
elapsed AND every incident that began by t + H has been delivered by the annotation feed.
"""

from __future__ import annotations

import numpy as np

from tdam.data.schema import Incident

OK, ACTIVE, RECOVERY, CENSORED = 0, 1, 2, 3
STATUS_NAMES = {OK: "ok", ACTIVE: "active", RECOVERY: "recovery", CENSORED: "censored"}


def label_windows(
    ent: np.ndarray,
    t_s: np.ndarray,
    incidents: list[Incident],
    entities: list[str],
    horizon_s: int,
    recovery_s: int,
    stream_end_s: int,
):
    """Vectorised labels for decision rows (ent index, time in seconds)."""
    n = len(t_s)
    y = np.zeros(n, dtype=np.int8)
    status = np.zeros(n, dtype=np.int8)
    index = {e: i for i, e in enumerate(entities)}
    by_ent: dict[int, list[Incident]] = {}
    for inc in incidents:
        if inc.entity in index:
            by_ent.setdefault(index[inc.entity], []).append(inc)
    for e, incs in by_ent.items():
        incs = sorted(incs, key=lambda i: (i.onset, i.incident_id))
        onsets = np.array([i.onset for i in incs])
        ends = np.array([i.end for i in incs])
        max_end = np.maximum.accumulate(ends)
        rows = np.nonzero(ent == e)[0]
        t = t_s[rows]
        nxt = np.searchsorted(onsets, t, side="right")  # first onset strictly after t
        has_next = nxt < len(onsets)
        next_onset = np.where(has_next, onsets[np.minimum(nxt, len(onsets) - 1)], np.iinfo(np.int64).max)
        y[rows] = (next_onset <= t + horizon_s).astype(np.int8)
        prev = nxt - 1
        have_prev = prev >= 0
        me = np.where(have_prev, max_end[np.maximum(prev, 0)], -1)
        active = have_prev & (me > t)
        recovery = have_prev & ~active & (me + recovery_s > t)
        status[rows[recovery]] = RECOVERY
        status[rows[active]] = ACTIVE
    status[t_s + horizon_s > stream_end_s] = CENSORED
    known_at = known_at_times(t_s, incidents, horizon_s)
    return y, status, known_at


def known_at_times(t_s: np.ndarray, incidents: list[Incident], horizon_s: int) -> np.ndarray:
    horizon_end = t_s + horizon_s
    if not incidents:
        return horizon_end.astype(np.int64)
    incs = sorted(incidents, key=lambda i: i.onset)
    onsets = np.array([i.onset for i in incs])
    avail = np.maximum.accumulate(np.array([i.label_available for i in incs]))
    j = np.searchsorted(onsets, horizon_end, side="right") - 1
    delivered = np.where(j >= 0, avail[np.maximum(j, 0)], -1)
    return np.maximum(horizon_end, delivered).astype(np.int64)
