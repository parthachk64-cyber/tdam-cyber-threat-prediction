"""Clean-environment CPU integration: prepare data, fit a real reduced LSTM + tree, predict, match
warnings, reload the bundle, replay, and generate the report."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from tdam.cli import main
from tdam.config import Config
from tdam.data.dataset import load_dir
from tdam.data.fixture import write_fixture
from tdam.data.replay import Replayer, replay
from tdam.evaluation.alerting import select_budget_threshold
from tdam.evaluation.experiment import run_experiment
from tdam.models.bundle import MODEL_NAMES, Bundle
from tdam.report import write_report
from tdam.temporal.splits import assign_incidents, make_partitions
from tdam.temporal.windows import build_windowset
from tests.conftest import SMALL

pytestmark = pytest.mark.slow


def tiny_cfg() -> Config:
    cfg = Config()
    cfg.model.epochs, cfg.model.oof_epochs, cfg.model.tree_iter, cfg.model.oof_folds = 2, 1, 30, 2
    cfg.model.threads = 2
    cfg.adaptive.update_every_minutes, cfg.adaptive.finetune_epochs = 480, 1
    return cfg


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("integration")
    write_fixture(SMALL, root / "data")
    cfg = tiny_cfg()
    ds = load_dir(root / "data")
    out = root / "run"
    result = run_experiment(ds, cfg, out, root)
    return {"root": root, "ds": ds, "cfg": cfg, "out": out, "result": result}


def test_artifacts_written_and_provenance_recorded(run):
    out = run["out"]
    for name in (
        "config_resolved.yaml",
        "run_meta.json",
        "metrics.json",
        "predictions_test.csv.gz",
        "incident_ledger.csv",
        "alerts.csv",
        "partitions.json",
        "latency.json",
        "adaptive_updates.json",
    ):
        assert (out / name).exists(), name
    meta = json.loads((out / "run_meta.json").read_text())
    for key in ("git_revision", "runtime", "seed", "feature_schema", "thresholds", "split_ids", "model_version", "dataset"):
        assert key in meta
    assert meta["dataset"]["source_file_sha256"]["events.jsonl"]
    assert meta["feature_schema"]["absent_modalities"] == ["dns", "flow", "process"]


def test_run_without_timely_warnings_succeeds_with_zero_early_recall(run):
    c = run["result"]["metrics"]["prevalence"]["budget"]["counts"]
    assert c["early"] == 0 and c["missed"] == c["incidents"] and c["alerts"] == 0


def test_bundle_save_load_prediction_parity(run):
    ws = build_windowset(run["ds"], run["cfg"])
    sub = ws.subset(np.arange(len(ws)) % 97 == 0)
    a = Bundle.load(run["out"] / "bundle").predict(sub.X, sub.lengths)
    b = Bundle.load(run["out"] / "bundle").predict(sub.X, sub.lengths)
    for m in MODEL_NAMES:
        assert np.allclose(a[m], b[m])
    meta = json.loads((run["out"] / "bundle" / "meta.json").read_text())
    assert meta["class_order"]["1"].startswith("annotated onset")
    saved = json.loads((run["out"] / "run_meta.json").read_text())["thresholds"]
    assert Bundle.load(run["out"] / "bundle").thresholds == saved


def test_replay_matches_batch_scores(run):
    lat = run["result"]["latency"]
    assert max(lat["parity_max_abs_diff"].values()) < 1e-5 and lat["parity_compared_rows"] > 0


def test_replay_duplicates_stale_late_and_bounded_state(run):
    ds, bundle = run["ds"], Bundle.load(run["out"] / "bundle")
    events = ds.events
    rep = Replayer(ds.entities, ds.timeline, 30, ds.cover)
    ev = events[len(events) // 2]
    t = int(ds.timeline.bin_of(ev.arrival_time)) + 1
    rep.ingest(ev, t)
    rep.ingest(ev, t)
    assert rep.stats["duplicates"] == 1 and rep.stats["ingested"] == 1
    stale = dataclasses.replace(ev, event_id="old", event_time=ev.event_time - 3600)
    rep.ingest(stale, t)
    assert rep.stats["stale_dropped"] == 1
    rep.prune(t + 1000)
    assert not rep.state, "idle entity state must expire"

    start = ds.timeline.n_bins // 2
    ticks = np.arange(start, start + 60)
    preds, stats = replay(events, ds.entities, ds.timeline, ds.cover, bundle, ticks)
    assert stats["duplicates"] > 0
    cut = ds.timeline.time_of(start + 30)
    prefix = [e for e in events if e.arrival_time <= cut]
    p_prefix, _ = replay(prefix, ds.entities, ds.timeline, ds.cover, bundle, ticks[:31])
    full = {(r["entity"], r["t"]): r["ensemble"] for r in preds}
    for r in p_prefix:
        assert abs(full[(r["entity"], r["t"])] - r["ensemble"]) < 1e-9, "future records changed an emitted prediction"


def test_alert_times_are_decision_times(run):
    alerts = pd.read_csv(run["out"] / "alerts.csv")
    preds = pd.read_csv(run["out"] / "predictions_test.csv.gz")
    a = alerts[(alerts["variant"] == "ensemble") & (alerts["operating_point"] == "budget")]
    stamped = set(zip(preds["entity"], preds["time_utc"], strict=True))
    assert all((e, t) in stamped for e, t in zip(a["entity"], a["alert_time"], strict=True))


def test_thresholds_are_independent_of_test_scores(run):
    ds, cfg = run["ds"], run["cfg"]
    ws = build_windowset(ds, cfg)
    bundle = Bundle.load(run["out"] / "bundle")
    sc = bundle.predict(ws.X, ws.lengths, ws.tab)["ensemble"]
    parts = make_partitions(ds.timeline.n_bins, dataclasses.asdict(cfg.splits))
    assigned = assign_incidents(ds.incidents, parts, ds.timeline, 30, 5)
    base = select_budget_threshold(ws, sc, parts["calib"], assigned["calib"], ds.entities, cfg)
    tampered = sc.copy()
    tampered[ws.t >= parts["test"].start] = 1.0
    assert select_budget_threshold(ws, tampered, parts["calib"], assigned["calib"], ds.entities, cfg) == base
    assert base == pytest.approx(bundle.thresholds["ensemble"]["budget"])


def test_adaptive_updates_use_only_matured_labels(run):
    log = json.loads((run["out"] / "adaptive_updates.json").read_text())
    assert len(log) >= 2
    for row in log[1:]:
        if row["max_known_at_used"] is not None:
            assert row["max_known_at_used"] <= row["update_time"], "an update used an annotation delivered after it was due"
    assert {row["version"] for row in log} == set(range(len(log)))


def test_report_and_predict_cli(run, tmp_path):
    md = tmp_path / "results.md"
    text = write_report(run["out"], md, "references/reported/paper_claims.yaml")
    assert "Same-alert-budget comparison" in text and "Published claims vs measurements" in text
    assert (run["out"] / "figures" / "pr_curve.png").exists() and (run["out"] / "traces" / "index.json").exists()
    sample = tmp_path / "sample.jsonl"
    lines = (run["root"] / "data" / "events.jsonl").read_text(encoding="utf-8").splitlines()[:4000]
    sample.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "pred.csv"
    args = ["predict", "--bundle", str(run["out"] / "bundle"), "--events", str(sample), "--out", str(out), "--stride", "5"]
    assert main(args) == 0
    assert "alert" in out.read_text().splitlines()[0]


def test_cli_data_commands(tmp_path):
    assert main(["data", "generate", "--out", str(tmp_path / "g"), "--hosts", "3", "--days", "1", "--seed", "5"]) == 0
    assert main(["data", "verify", str(tmp_path / "g")]) in (0, 1)
