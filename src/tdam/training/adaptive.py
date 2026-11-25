"""Walk-forward (prequential) adaptation.

Score first, update later. At each scheduled update time u the learner may use only windows whose
label was settled by u (`known_at <= u`: horizon elapsed and every incident that began by t + H
already delivered by the annotation feed). The scaler, blend weight and thresholds stay frozen;
the tree is refit on train + matured windows and the LSTM is fine-tuned on matured windows plus a
replay sample of training windows. Every window is scored by the model version that existed at its
decision time, and each version is logged.
"""

from __future__ import annotations

import copy

import numpy as np

from tdam.config import Config
from tdam.models.bundle import Bundle
from tdam.models.ensemble import blend
from tdam.models.lstm import LSTMClassifier, seed_everything
from tdam.models.tabular import TreeModel
from tdam.temporal.splits import Partition, partition_mask
from tdam.temporal.windows import WindowSet
from tdam.training.pipeline import select_training


def walk_forward(ws: WindowSet, bundle: Bundle, parts: dict[str, Partition], cfg: Config, timeline, seed: int):
    w, h = cfg.protocol.lookback_bins, cfg.protocol.horizon_minutes * 60 // cfg.protocol.bin_seconds
    test_idx = np.nonzero(partition_mask(ws.t, parts["test"], w, h))[0]
    train_idx = select_training(ws, partition_mask(ws.t, parts["train"], w, h) | partition_mask(ws.t, parts["val"], w, h), cfg.model.train_neg_stride)
    step = max(cfg.adaptive.update_every_minutes * 60 // cfg.protocol.bin_seconds, 1)
    first = int(ws.t[test_idx].min())
    last = int(ws.t[test_idx].max())

    lstm = copy.deepcopy(bundle.lstm)
    tree = bundle.tree
    scores = {k: np.zeros(len(ws)) for k in ("lstm", "tree", "ensemble")}
    version = np.zeros(len(ws), dtype=np.int32)
    log = [{"version": 0, "effective_from": int(first), "n_train": int(len(train_idx)), "matured_used": 0, "max_known_at_used": None}]
    rng = np.random.default_rng(seed)
    v = 0
    edge = first
    while edge <= last:
        nxt = edge + step
        blk = test_idx[(ws.t[test_idx] >= edge) & (ws.t[test_idx] < nxt)]
        if len(blk):
            sub = ws.subset(blk)
            p_l = lstm.predict_proba(bundle.scaler.transform(sub.X), sub.lengths)
            p_t = tree.predict_proba(sub.tab)
            scores["lstm"][blk], scores["tree"][blk] = p_l, p_t
            scores["ensemble"][blk] = blend(p_l, p_t, bundle.blend_weight)
            version[blk] = v
        if nxt > last:
            break
        now_s = timeline.time_of(nxt)
        matured = select_training(ws, np.isin(np.arange(len(ws)), test_idx) & (ws.known_at <= now_s), cfg.model.train_neg_stride)
        v += 1
        used = np.concatenate([train_idx, matured])
        if len(matured) and ws.y[used].sum() > 0:
            tree = TreeModel(seed + v, cfg.model.tree_iter, cfg.model.tree_lr, cfg.model.tree_depth).fit(ws.tab[used], ws.y[used])
            n_rep = min(len(train_idx), int(len(matured) * cfg.adaptive.replay_fraction) + 1)
            replay = rng.choice(train_idx, size=n_rep, replace=False)
            ft = np.concatenate([matured, replay])
            seed_everything(seed + v, cfg.model.threads)
            lstm_new = LSTMClassifier(bundle.lstm.n_features, bundle.lstm.hidden, bundle.lstm.layers)
            lstm_new.load_state(lstm.state())
            lstm_new.fit(
                bundle.scaler.transform(ws.X[ft]),
                ws.lengths[ft],
                ws.y[ft],
                epochs=cfg.adaptive.finetune_epochs,
                lr=cfg.model.lr * 0.5,
                batch=cfg.model.batch,
                seed=seed + v,
            )
            lstm = lstm_new
        log.append(
            {
                "version": v,
                "effective_from": int(nxt),
                "update_time": int(now_s),
                "n_train": int(len(used)),
                "matured_used": int(len(matured)),
                "max_known_at_used": int(ws.known_at[matured].max()) if len(matured) else None,
                "refit": bool(len(matured) and ws.y[used].sum() > 0),
            }
        )
        edge = nxt
    return scores, version, log
