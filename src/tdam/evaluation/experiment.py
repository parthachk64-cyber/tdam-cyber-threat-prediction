"""End-to-end experiment: fit -> frozen test -> adaptive test -> diagnostics -> artifacts.

Every number in the generated report is read back from the artifacts written here.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from tdam import __version__
from tdam.config import Config, dump_config
from tdam.data.dataset import Dataset
from tdam.data.io import sha256_file, write_json
from tdam.data.replay import replay
from tdam.evaluation.alerting import evaluate_alerts
from tdam.evaluation.latency import Timer, hardware_info
from tdam.evaluation.metrics import lead_time_summary, window_metrics
from tdam.models.bundle import MODEL_NAMES
from tdam.temporal.splits import partition_mask
from tdam.temporal.windows import build_windowset
from tdam.timeutil import iso
from tdam.training.adaptive import walk_forward
from tdam.training.pipeline import fit_bundle

LEARNED = ("lstm", "tree", "ensemble")


def git_revision(path: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "no-commit-yet"
    except (OSError, subprocess.SubprocessError):
        return "git-unavailable"


def shuffle_history(ws, seed: int):
    """Permute the time order of bins inside each window; labels stay fixed, no future data used."""
    rng = np.random.default_rng(seed)
    X = ws.X.copy()
    n, w, _ = X.shape
    perm = np.argsort(rng.random((n, w)), axis=1)
    X = np.take_along_axis(X, perm[..., None], axis=1)
    return ws.with_X(X)


def evaluate_variant(ws, scores, thresholds, parts, assigned, ds, cfg, test_mask) -> dict:
    out = {}
    usable = test_mask & ws.usable
    for op in ("budget", "f1"):
        thr = thresholds[op]
        ev = evaluate_alerts(ws, scores, thr, parts["test"], assigned["test"], ds.entities, cfg)
        res = ev["result"]
        leads = [r.lead_min for r in res.ledger if r.status == "early"]
        delays = [r.delay_min for r in res.ledger if r.status == "late"]
        out[op] = {
            "threshold": float(thr),
            "counts": ev["counts"],
            "lead_time_min": lead_time_summary(leads),
            "late_delay_min": lead_time_summary(delays),
            "fraction_warned_before_onset": (ev["counts"]["early"] / ev["counts"]["incidents"]) if ev["counts"]["incidents"] else None,
            "window": window_metrics(ws.y[usable], scores[usable], thr),
            "_ledger": [r.as_dict() for r in res.ledger],
            "_alerts": ev["alerts"],
        }
    out["window_ranking"] = {
        k: v for k, v in window_metrics(ws.y[usable], scores[usable], 0.5).items() if k in ("n", "positives", "prevalence", "average_precision", "roc_auc")
    }
    return out


def run_experiment(ds: Dataset, cfg: Config, out_dir: str | Path, repo_root: str | Path = ".", log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, h = cfg.protocol.lookback_bins, cfg.protocol.horizon_minutes * 60 // cfg.protocol.bin_seconds
    log(f"building windows ({len(ds.entities)} entities, {ds.timeline.n_bins} bins)")
    ws = build_windowset(ds, cfg)
    bundle, info, parts, assigned = fit_bundle(ds, ws, cfg, log=log)
    test_mask = partition_mask(ws.t, parts["test"], w, h)

    log("scoring frozen models")
    scores = bundle.predict(ws.X, ws.lengths, ws.tab)
    variants: dict[str, tuple[np.ndarray, str]] = {m: (scores[m], m) for m in MODEL_NAMES}

    log("walk-forward adaptive evaluation")
    ad_scores, ad_version, ad_log = walk_forward(ws, bundle, parts, cfg, ds.timeline, cfg.seed)
    for m in LEARNED:
        variants[f"{m}_adaptive"] = (ad_scores[m], m)

    log("order-shuffled history diagnostic")
    shuf = shuffle_history(ws.subset(test_mask), cfg.seed + 101)
    sh_scores = bundle.predict(shuf.X, shuf.lengths)
    for m in LEARNED:
        full = np.zeros(len(ws))
        full[test_mask] = sh_scores[m]
        variants[f"{m}_shuffled"] = (full, m)

    log("evaluating operating points on the test partition")
    metrics = {}
    for name, (sc, base) in variants.items():
        metrics[name] = evaluate_variant(ws, sc, bundle.thresholds[base], parts, assigned, ds, cfg, test_mask)

    log("replay parity and processing latency")
    latency = replay_check(ds, bundle, ws, scores, parts["test"], cfg, w)

    log("writing artifacts")
    write_artifacts(out, ds, cfg, ws, bundle, info, parts, assigned, variants, ad_version, ad_log, metrics, latency, test_mask, repo_root)
    return {"metrics": metrics, "info": info, "latency": latency, "out": str(out)}


def replay_check(ds, bundle, ws, scores, test_part, cfg, w, n_ticks: int = 240) -> dict:
    first = test_part.first_t(w)
    ticks = np.arange(first, first + n_ticks)
    timer = Timer()
    preds, stats = replay(ds.events or [], ds.entities, ds.timeline, ds.cover, bundle, ticks, timer)
    if not preds:
        return {"parity": None, "replay_stats": stats}
    index = {(ds.entities[int(e)], int(t)): i for i, (e, t) in enumerate(zip(ws.ent, ws.t, strict=True))}
    diffs = {m: 0.0 for m in MODEL_NAMES}
    matched = 0
    for row in preds:
        i = index.get((row["entity"], row["t"]))
        if i is None:
            continue
        matched += 1
        for m in MODEL_NAMES:
            diffs[m] = max(diffs[m], abs(row[m] - float(scores[m][i])))
    n_ent = len(ds.entities)
    timer_sum = timer.summary(warmup=20, batch_size=n_ent)
    e2e = [a + b + c for a, b, c in zip(timer.samples["ingest_ms"], timer.samples["feature_ms"], timer.samples["score_ms"], strict=True)]
    arr = np.asarray(e2e[20:])
    timer_sum["end_to_end_ms"] = {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "n_measured": int(len(arr)),
        "n_warmup_discarded": 20,
        "batch_size": n_ent,
    }
    return {
        "parity_max_abs_diff": diffs,
        "parity_compared_rows": matched,
        "ticks": int(n_ticks),
        "replay_stats": stats,
        "latency": timer_sum,
        "hardware": hardware_info(),
        "note": "processing milliseconds only; unrelated to detection delay or advance-warning minutes",
    }


def write_artifacts(out, ds, cfg, ws, bundle, info, parts, assigned, variants, ad_version, ad_log, metrics, latency, test_mask, repo_root):
    dump_config(cfg, out / "config_resolved.yaml")
    write_json(out / "partitions.json", info["partitions"])
    write_json(out / "adaptive_updates.json", [{**r, **({"update_time_utc": iso(r["update_time"])} if "update_time" in r else {})} for r in ad_log])
    write_json(out / "latency.json", latency)
    bundle.save(out / "bundle")

    idx = np.nonzero(test_mask)[0]
    frame = pd.DataFrame(
        {
            "time_utc": [iso(int(s)) for s in ws.t_s[idx]],
            "entity": [ds.entities[int(e)] for e in ws.ent[idx]],
            "t_bin": ws.t[idx],
            "status": [("ok", "active", "recovery", "censored")[int(s)] for s in ws.status[idx]],
            "y": ws.y[idx],
        }
    )
    for name in ("prevalence", "logistic", "tree", "lstm", "ensemble", "lstm_adaptive", "tree_adaptive", "ensemble_adaptive"):
        frame[f"score_{name}"] = np.round(variants[name][0][idx], 6)
    frame["adaptive_model_version"] = ad_version[idx]
    frame.to_csv(out / "predictions_test.csv.gz", index=False, compression="gzip")

    ledgers, alerts = [], []
    slim = {}
    for name, per_op in metrics.items():
        slim[name] = {}
        for op in ("budget", "f1"):
            blk = dict(per_op[op])
            for row in blk.pop("_ledger"):
                ledgers.append({"variant": name, "operating_point": op, **row})
            for ent, t in blk.pop("_alerts"):
                alerts.append({"variant": name, "operating_point": op, "entity": ent, "alert_time": iso(t)})
            slim[name][op] = blk
        slim[name]["window_ranking"] = per_op["window_ranking"]
    pd.DataFrame(ledgers).to_csv(out / "incident_ledger.csv", index=False)
    pd.DataFrame(alerts, columns=["variant", "operating_point", "entity", "alert_time"]).to_csv(out / "alerts.csv", index=False)
    write_json(out / "metrics.json", slim)

    files = {}
    for name in ("events.jsonl", "incidents.jsonl", "coverage.json"):
        p = Path(ds.meta.get("source_dir", "")) / name
        if p.exists():
            files[name] = sha256_file(p)
    meta = {
        "dataset": {"name": ds.name, "source": ds.source, "source_file_sha256": files, "ingest_stats": ds.meta.get("ingest_stats")},
        "git_revision": git_revision(Path(repo_root)),
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "tdam": __version__},
        "seed": cfg.seed,
        "model_version": bundle.version,
        "blend_weight": bundle.blend_weight,
        "thresholds": bundle.thresholds,
        "split_ids": info["partitions"]["partitions"],
        "feature_schema": json.loads((out / "bundle" / "meta.json").read_text(encoding="utf-8"))["feature_schema"],
        "oof": {"folds": info["oof_folds"], "blend_detail": info["blend_detail"]},
        "train": {"windows": info["train_windows"], "positive_windows": info["train_positive_windows"]},
        "incidents_per_partition": info["incidents_per_partition"],
        "provenance": cfg.provenance,
    }
    write_json(out / "run_meta.json", meta)
