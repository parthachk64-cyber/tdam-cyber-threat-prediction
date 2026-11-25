"""Window sets: every (entity, decision time) the defender could have scored."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tdam.config import Config
from tdam.data.dataset import Dataset
from tdam.temporal.binning import build_entity_windows
from tdam.temporal.features import tab_features
from tdam.temporal.labels import OK, label_windows


@dataclass
class WindowSet:
    ent: np.ndarray
    t: np.ndarray  # decision bin
    t_s: np.ndarray  # decision time, epoch seconds
    X: np.ndarray  # [N, W, 6]
    lengths: np.ndarray
    y: np.ndarray
    status: np.ndarray
    known_at: np.ndarray
    _tab: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def tab(self) -> np.ndarray:
        if self._tab is None:
            self._tab = tab_features(self.X)
        return self._tab

    def subset(self, mask: np.ndarray) -> "WindowSet":
        return WindowSet(
            self.ent[mask],
            self.t[mask],
            self.t_s[mask],
            self.X[mask],
            self.lengths[mask],
            self.y[mask],
            self.status[mask],
            self.known_at[mask],
            None if self._tab is None else self._tab[mask],
        )

    def with_X(self, X: np.ndarray) -> "WindowSet":
        return WindowSet(self.ent, self.t, self.t_s, X, self.lengths, self.y, self.status, self.known_at)

    @property
    def usable(self) -> np.ndarray:
        """Windows with a settled, eligible label (status OK)."""
        return self.status == OK


def build_windowset(ds: Dataset, cfg: Config, stride: int | None = None, incidents=None) -> WindowSet:
    w = cfg.protocol.lookback_bins
    stride = stride or cfg.protocol.decision_stride
    tl = ds.timeline
    parts = []
    for e in range(len(ds.entities)):
        dec = np.nonzero(ds.cover[e] & (np.arange(tl.n_bins) % stride == 0))[0]
        if not len(dec):
            continue
        X, lengths = build_entity_windows(ds.rows, e, tl, w, ds.cover[e], dec)
        parts.append((np.full(len(dec), e), dec, X, lengths))
    ent = np.concatenate([p[0] for p in parts])
    t = np.concatenate([p[1] for p in parts])
    X = np.concatenate([p[2] for p in parts])
    lengths = np.concatenate([p[3] for p in parts])
    order = np.lexsort((ent, t))
    ent, t, X, lengths = ent[order], t[order], X[order], lengths[order]
    t_s = tl.time_of(t.astype(np.int64))
    y, status, known = label_windows(
        ent,
        t_s,
        ds.incidents if incidents is None else incidents,
        ds.entities,
        cfg.horizon_s,
        cfg.protocol.recovery_minutes * 60,
        tl.end,
    )
    return WindowSet(ent, t, t_s, X, lengths, y, status, known)
