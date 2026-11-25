"""Generate docs/results.md, figures and per-incident traces from saved run artifacts only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

MAIN = ("prevalence", "logistic", "tree", "lstm", "ensemble")
ADAPT = ("lstm", "tree", "ensemble")


def _f(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _load(run: Path):
    return (
        json.loads((run / "metrics.json").read_text(encoding="utf-8")),
        json.loads((run / "run_meta.json").read_text(encoding="utf-8")),
        json.loads((run / "latency.json").read_text(encoding="utf-8")),
        pd.read_csv(run / "predictions_test.csv.gz"),
        pd.read_csv(run / "incident_ledger.csv"),
        pd.read_csv(run / "alerts.csv"),
    )


def _row(name, v, op="budget"):
    c, lead = v[op]["counts"], v[op]["lead_time_min"]
    q = f"{_f(lead['median'], 1)} ({_f(lead['q1'], 1)}-{_f(lead['q3'], 1)}), n={lead['n']}" if lead["n"] else "none"
    rk = v["window_ranking"]
    return (
        f"| {name} | {c['early']}/{c['incidents']} | {c['late_only']} | {c['missed']} | {c['unmatched_alerts']} "
        f"| {_f(c['unmatched_per_host_hour'], 4)} | {q} | {_f(rk['average_precision'])} | {_f(rk['roc_auc'])} | {_f(v[op]['threshold'], 4)} |"
    )


HEAD = "| model | advance warnings (early / incidents) | late detections (late-only) | missed | false alarms (unmatched alerts) | unmatched / host-hour | lead min, median (IQR) | AP | ROC-AUC | threshold |\n|---|---|---|---|---|---|---|---|---|---|"


def figures(run: Path, metrics, preds, ledger) -> None:
    figdir = run / "figures"
    figdir.mkdir(exist_ok=True)
    ok = preds[preds["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in MAIN:
        if ok["y"].nunique() < 2:
            break
        p, r, _ = precision_recall_curve(ok["y"], ok[f"score_{m}"])
        ax.plot(r, p, label=f"{m} (AP {metrics[m]['window_ranking']['average_precision'] or 0:.3f})")
    ax.axhline(ok["y"].mean(), color="gray", ls=":", label="prevalence")
    ax.set_xlabel("recall (windows)")
    ax.set_ylabel("precision (windows)")
    ax.set_title("Precision-recall, test partition (synthetic fixture)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "pr_curve.png", dpi=130)
    plt.close(fig)

    led = ledger[(ledger["variant"] == "ensemble") & (ledger["operating_point"] == "budget")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [2, 1]})
    leads = led.loc[led["status"] == "early", "lead_min"]
    axes[0].hist(leads, bins=np.arange(0.5, 6.5, 1.0), color="#2a6f97")
    axes[0].set_xlabel("advance-warning lead (min), timely warnings only")
    axes[0].set_ylabel("incidents")
    axes[0].set_title(f"Lead time, ensemble @ budget (n={len(leads)})")
    counts = led["status"].value_counts().reindex(["early", "late", "missed"]).fillna(0)
    axes[1].bar(counts.index, counts.values, color=["#2a6f97", "#e9a03b", "#b23a48"])
    axes[1].set_title("Outcome per incident")
    fig.tight_layout()
    fig.savefig(figdir / "lead_time.png", dpi=130)
    plt.close(fig)


def traces(run: Path, meta, preds, ledger, alerts, horizon_min=5, pad_min=25) -> dict:
    """One real trace each for an early, a late and a missed incident (ensemble @ budget)."""
    thr = meta["thresholds"]["ensemble"]["budget"]
    led = ledger[(ledger["variant"] == "ensemble") & (ledger["operating_point"] == "budget")]
    al = alerts[(alerts["variant"] == "ensemble") & (alerts["operating_point"] == "budget")]
    tdir = run / "traces"
    tdir.mkdir(exist_ok=True)
    chosen = {}
    for status in ("early", "late", "missed"):
        rows = led[led["status"] == status]
        if rows.empty:
            continue
        r = rows.iloc[0]
        onset = pd.Timestamp(r["onset"])
        ent = r["entity"]
        p = preds[preds["entity"] == ent].copy()
        p["ts"] = pd.to_datetime(p["time_utc"])
        win = p[(p["ts"] >= onset - pd.Timedelta(minutes=pad_min)) & (p["ts"] <= onset + pd.Timedelta(minutes=pad_min))].copy()
        a_times = pd.to_datetime(al.loc[al["entity"] == ent, "alert_time"])
        win["alert_issued"] = win["ts"].isin(set(a_times)).astype(int)
        win["threshold"] = thr
        win["onset_utc"] = r["onset"]
        win["incident_id"] = r["incident_id"]
        win = win[["time_utc", "entity", "status", "y", "score_ensemble", "threshold", "alert_issued", "onset_utc", "incident_id"]]
        win.to_csv(tdir / f"trace_{status}_{r['incident_id']}.csv", index=False)
        chosen[status] = {"incident_id": r["incident_id"], "entity": ent, "onset": r["onset"], "lead_min": float(r["lead_min"]), "delay_min": None if pd.isna(r["delay_min"]) else float(r["delay_min"]), "file": f"traces/trace_{status}_{r['incident_id']}.csv"}
        chosen[status]["_win"] = win
    if chosen:
        fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (status, info) in zip(axes, chosen.items(), strict=True):
            w = info["_win"]
            ts = pd.to_datetime(w["time_utc"])
            ax.plot(ts, w["score_ensemble"], color="#2a6f97", lw=1.2, label="ensemble score")
            ax.axhline(thr, color="gray", ls="--", lw=1, label="threshold")
            ax.axvline(pd.Timestamp(info["onset"]), color="#b23a48", lw=1.2, label="incident onset")
            fired = w[w["alert_issued"] == 1]
            ax.scatter(pd.to_datetime(fired["time_utc"]), fired["score_ensemble"], color="#e9a03b", zorder=3, label="alert issued")
            ax.set_title(f"{status}: {info['incident_id']}", fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=7)
        axes[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(run / "figures" / "traces.png", dpi=130)
        plt.close(fig)
    return {k: {a: b for a, b in v.items() if a != "_win"} for k, v in chosen.items()}


def write_report(run: str | Path, out_md: str | Path, claims_yaml: str | Path | None = None) -> str:
    run = Path(run)
    metrics, meta, latency, preds, ledger, alerts = _load(run)
    figures(run, metrics, preds, ledger)
    tr = traces(run, meta, preds, ledger, alerts)
    (run / "traces" / "index.json").write_text(json.dumps(tr, indent=2), encoding="utf-8")
    ds = meta["dataset"]
    inc = meta["incidents_per_partition"]
    budget = yaml.safe_load((run / "config_resolved.yaml").read_text(encoding="utf-8"))["alerting"]["budget_per_host_hour"]
    lines = [
        "# Results (generated)",
        "",
        "> Generated by `tdam report` from the saved artifacts in this folder. Do not edit by hand.",
        "",
        "## Data source for every measurement below",
        "",
        f"- **Dataset:** `{ds['name']}`, source label **{ds['source']}**: output of the repository's seeded generator (seed {meta['seed']}). This is a *demonstration* on synthetic telemetry. It is not the paper's data and not an independent benchmark, so nothing here reproduces or contradicts the paper's reported results.",
        f"- **Partitions:** chronological train / val / calib / test; incidents per partition {inc}. All tables below are on the **test** partition. Thresholds were chosen on **calib** only.",
        f"- **Alert budget:** at most {budget} unmatched warnings per monitored host-hour on calib; every model gets its own calib-selected threshold at the same budget (`config_resolved.yaml`).",
        f"- **Software:** {meta['runtime']}; git revision `{meta['git_revision']}`.",
        "",
        "## 1. Same-alert-budget comparison (test partition)",
        "",
        "Early = a warning issued before the annotated onset and within H. Late-only = first credited alert came after onset. Missed = no credited alert. False alarms = alerts matched to no annotated incident (*unmatched to known labels*; annotations are incomplete, so this is not proof of a false positive). Source: `metrics.json`, `incident_ledger.csv`, `alerts.csv`.",
        "",
        HEAD,
    ]
    lines += [_row(m, metrics[m]) for m in MAIN]
    lines += ["", "## 2. Validation-F1 operating threshold (test partition)", "", HEAD]
    lines += [_row(m, metrics[m], "f1") for m in MAIN]
    lines += ["", "## 3. Frozen vs adaptive vs order-shuffled history (same thresholds, same budget)", "", HEAD]
    for m in ADAPT:
        for suffix, label in (("", "frozen"), ("_adaptive", "walk-forward"), ("_shuffled", "shuffled history")):
            lines.append(_row(f"{m} / {label}", metrics[m + suffix]))
    lines += ["", "## 4. Window-level metrics at the budget threshold", "", "| model | n | positives | prevalence | TP | FP | FN | TN | accuracy | always-negative accuracy | precision | recall | F1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MAIN:
        w = metrics[m]["budget"]["window"]
        lines.append(f"| {m} | {w['n']} | {w['positives']} | {_f(w['prevalence'], 4)} | {w['tp']} | {w['fp']} | {w['fn']} | {w['tn']} | {_f(w['accuracy'], 4)} | {_f(w['always_negative_accuracy'], 4)} | {_f(w['precision'])} | {_f(w['recall'])} | {_f(w['f1'])} |")
    lat = latency.get("latency", {})
    lines += ["", "## 5. Processing latency (milliseconds; NOT detection delay, NOT advance warning)", "", f"Replay of {latency.get('ticks')} decision ticks over saved telemetry in arrival order. Hardware: `{latency.get('hardware', {}).get('processor', '')}`, torch threads {latency.get('hardware', {}).get('torch_threads')}, CPU only, monotonic clock, 20 warm-up ticks discarded; batch = all monitored entities at a tick. Source: `latency.json`.", "", "| stage | p50 ms | p95 ms | p99 ms | measured ticks | batch (entities) |", "|---|---|---|---|---|---|"]
    for k, v in lat.items():
        lines.append(f"| {k} | {_f(v['p50_ms'], 2)} | {_f(v['p95_ms'], 2)} | {_f(v['p99_ms'], 2)} | {v['n_measured']} | {v['batch_size']} |")
    par = latency.get("parity_max_abs_diff", {})
    lines += ["", f"Replay-vs-batch score parity (max absolute difference over {latency.get('parity_compared_rows')} compared rows): `{par}`. Replay stats: `{latency.get('replay_stats')}`.", ""]
    lines += ["## 6. Per-incident traces (ensemble @ budget, real rows from `predictions_test.csv.gz`)", ""]
    for status, info in tr.items():
        extra = f"lead {info['lead_min']:.0f} min" if status == "early" else (f"detection delay {info['delay_min']:.0f} min after onset" if status == "late" else "no credited alert")
        lines.append(f"- **{status}** `{info['incident_id']}` onset {info['onset']}: {extra}. File: [`{info['file']}`]({info['file']}).")
    lines += ["", "![traces](figures/traces.png)", "", "![pr](figures/pr_curve.png)", "", "![lead](figures/lead_time.png)", ""]
    lines += negative_findings(metrics)
    if claims_yaml and Path(claims_yaml).exists():
        lines += claims_table(claims_yaml)
    text = "\n".join(lines) + "\n"
    Path(out_md).write_text(text, encoding="utf-8")
    return text


def negative_findings(metrics) -> list[str]:
    out = ["## 7. Findings, including unfavourable ones", ""]
    b = {m: metrics[m]["budget"]["counts"] for m in MAIN}
    best_early = max(MAIN, key=lambda m: (b[m]["early"], -b[m]["unmatched_alerts"]))
    out.append(f"- Most advance warnings at the shared budget: **{best_early}** ({b[best_early]['early']}/{b[best_early]['incidents']}).")
    if b["ensemble"]["early"] < max(b["tree"]["early"], b["lstm"]["early"]):
        out.append("- **The ensemble did not give the most advance warnings**; a single branch did. Blending optimised out-of-fold average precision, not early-warning count.")
    ap = {m: metrics[m]["window_ranking"]["average_precision"] for m in MAIN}
    base = metrics["prevalence"]["window_ranking"]["average_precision"]
    out.append(f"- Window average precision is low in absolute terms (best {max(v for v in ap.values() if v is not None):.3f} against prevalence {base:.4f}); most incidents are detected after onset, not before.")
    out.append(f"- Prevalence-only issues {b['prevalence']['alerts']} alerts at the budget threshold: a constant score carries no timing information.")
    for m in ADAPT:
        d = metrics[m + "_adaptive"]["window_ranking"]["average_precision"] - metrics[m]["window_ranking"]["average_precision"]
        out.append(f"- Walk-forward adaptation changed {m} window AP by {d:+.3f} (single seed; no claim of significance).")
    for m in ADAPT:
        d = metrics[m + "_shuffled"]["window_ranking"]["average_precision"] - metrics[m]["window_ranking"]["average_precision"]
        out.append(f"- Shuffling the order of bins inside each window changed {m} AP by {d:+.3f}: evidence that the models use temporal order on this fixture, which was constructed to contain it.")
    out.append("- Single seed, 25-incident test set, no confidence intervals: differences of a few incidents are within noise.")
    out.append("")
    return out


def claims_table(path) -> list[str]:
    claims = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = ["## 8. Published claims vs measurements obtained in this repository", "", "| claim (source) | published value | status | what this repository measured |", "|---|---|---|---|"]
    for c in claims["claims"]:
        out.append(f"| {c['claim']} ({c['location']}) | {c['value']} | {c['status']} | {c['measured_here']} |")
    out.append("")
    return out
