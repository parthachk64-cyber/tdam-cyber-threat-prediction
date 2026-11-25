"""Seeded training: base models, chronological out-of-fold blending weight, thresholds.

Partitions (chronological): train | val | calib | test.
  train  fit scaler, LSTM, tree, logistic, prevalence
  val    LSTM early stopping only
  oof    expanding-window folds INSIDE train: blend weight
  calib  operating thresholds (alert budget and F1) -- the only place thresholds are chosen
  test   never used to fit or choose anything
"""

from __future__ import annotations

import numpy as np

from tdam.config import Config
from tdam.data.dataset import Dataset
from tdam.evaluation.alerting import select_budget_threshold, select_f1_threshold
from tdam.models.bundle import Bundle
from tdam.models.ensemble import learn_weight
from tdam.models.lstm import LSTMClassifier, seed_everything
from tdam.models.tabular import LogisticModel, PrevalenceModel, TreeModel
from tdam.temporal.features import SeqScaler
from tdam.temporal.splits import assign_incidents, make_partitions, manifest, partition_mask, span_mask
from tdam.temporal.windows import WindowSet


def select_training(ws: WindowSet, mask: np.ndarray, neg_stride: int) -> np.ndarray:
    """Usable windows in `mask`; every positive is kept, negatives are thinned deterministically."""
    keep = mask & ws.usable & ((ws.y == 1) | (ws.t % max(neg_stride, 1) == 0))
    return np.nonzero(keep)[0]


def fit_lstm(scaler, ws, tr_idx, va_idx, cfg: Config, seed: int, epochs: int) -> LSTMClassifier:
    seed_everything(seed, cfg.model.threads)
    lstm = LSTMClassifier(ws.X.shape[-1], cfg.model.hidden, cfg.model.layers)
    tr, va = ws.subset(tr_idx), ws.subset(va_idx) if va_idx is not None and len(va_idx) else None
    lstm.fit(
        scaler.transform(tr.X),
        tr.lengths,
        tr.y,
        scaler.transform(va.X) if va is not None else None,
        va.lengths if va is not None else None,
        va.y if va is not None else None,
        epochs=epochs,
        patience=cfg.model.patience,
        lr=cfg.model.lr,
        batch=cfg.model.batch,
        seed=seed,
    )
    return lstm


def fit_tree(ws, idx, cfg: Config, seed: int) -> TreeModel:
    sub = ws.subset(idx)
    m = cfg.model
    return TreeModel(seed, m.tree_iter, m.tree_lr, m.tree_depth).fit(sub.tab, sub.y)


def oof_predictions(ws: WindowSet, train_end: int, cfg: Config, seed: int):
    """Expanding-window chronological folds inside the training span."""
    w, h = cfg.protocol.lookback_bins, cfg.protocol.horizon_minutes * 60 // cfg.protocol.bin_seconds
    k = cfg.model.oof_folds
    edges = [int(round(train_end * (i + 1) / (k + 1))) for i in range(k + 1)]
    ys, pl, pt, folds = [], [], [], []
    for f in range(1, k + 1):
        fit_end, pred_end = edges[f - 1], edges[f]
        v_start = int(fit_end * 0.8)
        tr = select_training(ws, span_mask(ws.t, 0, v_start, w, h), cfg.model.train_neg_stride)
        va = np.nonzero(span_mask(ws.t, v_start, fit_end, w, h) & ws.usable)[0]
        te = np.nonzero(span_mask(ws.t, fit_end, pred_end, w, h) & ws.usable)[0]
        if len(tr) == 0 or ws.y[tr].sum() == 0 or len(te) == 0:
            folds.append({"fold": f, "skipped": "no positives in fit span or empty prediction block"})
            continue
        scaler = SeqScaler().fit(ws.X[tr])
        lstm = fit_lstm(scaler, ws, tr, va, cfg, seed + f, cfg.model.oof_epochs)
        tree = fit_tree(ws, tr, cfg, seed + f)
        sub = ws.subset(te)
        pl.append(lstm.predict_proba(scaler.transform(sub.X), sub.lengths))
        pt.append(tree.predict_proba(sub.tab))
        ys.append(sub.y)
        folds.append({"fold": f, "fit_bins": [0, v_start], "predict_bins": [fit_end, pred_end], "n_pred": int(len(te)), "positives": int(sub.y.sum())})
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)  # noqa: E731
    return cat(ys).astype(int), cat(pl), cat(pt), folds


def fit_bundle(ds: Dataset, ws: WindowSet, cfg: Config, seed: int | None = None, log=print):
    """Returns (bundle, info, partitions, incidents_per_partition)."""
    seed = cfg.seed if seed is None else seed
    w, h = cfg.protocol.lookback_bins, cfg.protocol.horizon_minutes * 60 // cfg.protocol.bin_seconds
    parts = make_partitions(ds.timeline.n_bins, {"train": cfg.splits.train, "val": cfg.splits.val, "calib": cfg.splits.calib, "test": cfg.splits.test})
    assigned = assign_incidents(ds.incidents, parts, ds.timeline, w, cfg.protocol.horizon_minutes)

    log("fitting out-of-fold predictions inside the training span")
    y_o, p_l, p_t, folds = oof_predictions(ws, parts["train"].end, cfg, seed)
    weight, wdetail = learn_weight(p_l, p_t, y_o)

    log("fitting base models on the training partition")
    tr = select_training(ws, partition_mask(ws.t, parts["train"], w, h), cfg.model.train_neg_stride)
    va = np.nonzero(partition_mask(ws.t, parts["val"], w, h) & ws.usable)[0]
    if len(tr) == 0 or ws.y[tr].sum() == 0:
        raise ValueError("the training partition has no positive windows; the fixture or split is unusable")
    scaler = SeqScaler().fit(ws.X[tr])
    lstm = fit_lstm(scaler, ws, tr, va, cfg, seed, cfg.model.epochs)
    tree = fit_tree(ws, tr, cfg, seed)
    sub = ws.subset(tr)
    logistic = LogisticModel(seed).fit(sub.tab, sub.y)
    prevalence = PrevalenceModel().fit(None, sub.y)
    bundle = Bundle(scaler, lstm, tree, logistic, prevalence, weight, w, version="v0-frozen")

    log("choosing thresholds on the calibration partition")
    cal_scores = bundle.predict(ws.X, ws.lengths, ws.tab)
    thresholds = {}
    for name, sc in cal_scores.items():
        thresholds[name] = {
            "budget": select_budget_threshold(ws, sc, parts["calib"], assigned["calib"], ds.entities, cfg),
            "f1": select_f1_threshold(ws, sc, parts["calib"], cfg),
        }
    bundle.thresholds = thresholds
    info = {
        "seed": seed,
        "partitions": manifest(parts, assigned, ds.timeline, w, h),
        "train_windows": int(len(tr)),
        "train_positive_windows": int(ws.y[tr].sum()),
        "oof_folds": folds,
        "oof_windows": int(len(y_o)),
        "blend_weight": weight,
        "blend_detail": wdetail,
        "thresholds": thresholds,
        "incidents_per_partition": {k: len(v) for k, v in assigned.items()},
    }
    bundle.meta = {"seed": seed, "protocol": cfg.to_dict()["protocol"]}
    return bundle, info, parts, assigned
