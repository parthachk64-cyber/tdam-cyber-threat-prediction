"""Feature transforms. Only past-window quantities; nothing here sees labels or the future.

Channel order (X[..., c]): events, fail, success, uniq_dst, gap_min, observed.
`observed` = 1 where the source was monitored. A true zero count has observed = 1; an unknown
(unmonitored) bin has observed = 0 and zero counts. Absent modalities are listed in the feature
schema, never imputed.
"""

from __future__ import annotations

import numpy as np

from tdam.temporal.binning import CHANNELS

TAB_NAMES = (
    "total_events",
    "total_fail",
    "fail_rate",
    "total_uniq_dst",
    "max_bin_events",
    "std_bin_events",
    "events_last5",
    "events_prior",
    "events_ratio",
    "fail_rate_last10",
    "fail_rate_prior20",
    "fail_rate_delta",
    "uniq_dst_last10",
    "uniq_dst_slope",
    "gap_last_min",
    "observed_frac",
)


def tab_features(X: np.ndarray) -> np.ndarray:
    """[N, W, 6] -> [N, 16] window summaries incl. recent-vs-prior activity change."""
    n, f, nd = X[..., 0], X[..., 1], X[..., 3]
    w = X.shape[1]
    split5, split10 = w - 5, w - 10
    total = n.sum(1)
    tf = f.sum(1)
    last5 = n[:, split5:].mean(1)
    prior = n[:, :split5].mean(1)
    fr10 = f[:, split10:].sum(1) / (n[:, split10:].sum(1) + 1.0)
    fr20 = f[:, :split10].sum(1) / (n[:, :split10].sum(1) + 1.0)
    pos = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
    slope = (nd * pos[None, :]).sum(1) / (pos**2).sum()
    out = np.stack(
        [
            total,
            tf,
            tf / (total + 1.0),
            nd.sum(1),
            n.max(1),
            n.std(1),
            last5,
            prior,
            (last5 + 0.5) / (prior + 0.5),
            fr10,
            fr20,
            fr10 - fr20,
            nd[:, split10:].sum(1),
            slope,
            X[:, -1, 4],
            X[..., 5].mean(1),
        ],
        axis=1,
    )
    return out.astype(np.float32)


def seq_transform(X: np.ndarray) -> np.ndarray:
    """Variance-stabilise counts (log1p) and scale the gap. Shape preserved."""
    out = X.copy()
    out[..., :4] = np.log1p(out[..., :4])
    out[..., 4] = out[..., 4] / X.shape[1]
    return out


class SeqScaler:
    """Per-channel standardiser fitted on observed bins of the training partition only."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray, lengths: np.ndarray | None = None) -> "SeqScaler":
        Z = seq_transform(X)
        obs = X[..., 5] > 0
        if obs.sum() < 2:
            obs = np.ones_like(obs)
        c = Z.shape[-1]
        mean = np.zeros(c, dtype=np.float32)
        std = np.ones(c, dtype=np.float32)
        for k in range(c - 1):  # the mask channel stays 0/1
            v = Z[..., k][obs]
            mean[k] = v.mean()
            std[k] = max(float(v.std()), 1e-6)
        self.mean, self.std = mean, std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (seq_transform(X) - self.mean) / self.std
        Z[..., 5] = X[..., 5]
        # padded bins (all-zero) stay exactly zero so padding is inert
        pad = (X == 0).all(axis=-1)
        Z[pad] = 0.0
        return Z.astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, st: dict) -> "SeqScaler":
        s = cls()
        s.mean = np.asarray(st["mean"], dtype=np.float32)
        s.std = np.asarray(st["std"], dtype=np.float32)
        return s


def feature_schema(w: int) -> dict:
    return {
        "sequence": {"shape": ["batch", w, len(CHANNELS)], "channels": list(CHANNELS)},
        "tabular": {"names": list(TAB_NAMES)},
        "absent_modalities": ["dns", "flow", "process"],
        "note": "counts of zero with observed=1 are true zeros; observed=0 means unknown coverage",
    }
