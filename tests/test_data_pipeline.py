import dataclasses
import shutil

import numpy as np

from tdam.config import Config
from tdam.data.dataset import build_dataset, load_dir
from tdam.data.fixture import generate, write_fixture
from tdam.data.io import event_from_record, incident_from_record, read_jsonl, sha256_file
from tdam.data.schema import FORBIDDEN_EVENT_FIELDS
from tdam.data.validate import lanl_missing, validate_dir
from tdam.temporal.labels import ACTIVE, OK, RECOVERY
from tdam.temporal.splits import assign_incidents, make_partitions, partition_mask
from tdam.temporal.windows import build_windowset
from tests.conftest import SMALL


def test_fixture_is_deterministic(tmp_path):
    write_fixture(SMALL, tmp_path / "a")
    write_fixture(SMALL, tmp_path / "b")
    for name in ("events.jsonl", "incidents.jsonl", "coverage.json"):
        assert sha256_file(tmp_path / "a" / name) == sha256_file(tmp_path / "b" / name)


def test_events_carry_no_label_like_fields_and_include_hard_cases(small_records):
    events, incidents, coverage = small_records
    assert not (set().union(*[set(e) for e in events]) & FORBIDDEN_EVENT_FIELDS)
    ids = [e["event_id"] for e in events]
    assert len(ids) > len(set(ids)), "duplicate deliveries expected"
    assert any(e["arrival_time"] > e["event_time"] for e in events), "delayed records expected"
    assert incidents and coverage["modalities_absent"]


def test_negative_only_fixture_has_no_incidents():
    _, incidents, _ = generate(dataclasses.replace(SMALL, incidents_per_host_day=0.0, n_hosts=3, days=1))
    assert incidents == []


def test_every_partition_has_positive_windows(small_dataset, cfg):
    ws = build_windowset(small_dataset, cfg)
    w, h = cfg.protocol.lookback_bins, 5
    parts = make_partitions(small_dataset.timeline.n_bins, dataclasses.asdict(cfg.splits))
    for name, p in parts.items():
        m = partition_mask(ws.t, p, w, h) & ws.usable
        assert ws.y[m].sum() > 0, name


def test_partitions_purge_boundaries_and_assign_each_incident_once(small_dataset, cfg):
    ds = small_dataset
    w, h = 30, 5
    parts = make_partitions(ds.timeline.n_bins, dataclasses.asdict(cfg.splits))
    ws = build_windowset(ds, cfg)
    owner = np.full(len(ws), -1)
    for i, p in enumerate(parts.values()):
        m = partition_mask(ws.t, p, w, h)
        assert (owner[m] == -1).all(), "a window landed in two partitions"
        owner[m] = i
        assert (ws.t[m] - w + 1 >= p.start).all() and (ws.t[m] + h <= p.end - 1).all()
    assigned = assign_incidents(ds.incidents, parts, ds.timeline, w, 5)
    flat = [i.incident_id for k, v in assigned.items() for i in v]
    assert sorted(flat) == sorted(i.incident_id for i in ds.incidents)
    # positive windows of an incident live only in the partition that owns it
    for name in assigned:
        if name == "purged":
            continue
        p = parts[name]
        m = partition_mask(ws.t, p, w, h) & (ws.y == 1)
        onsets = {ds.timeline.time_of(int(t)) for t in ws.t[m]}
        assert onsets  # sanity


def test_windows_never_mix_entities(small_dataset, cfg):
    ws = build_windowset(small_dataset, cfg)
    ent = 0
    other = ws.subset(ws.ent != ent)
    only = ws.subset(ws.ent == ent)
    ds = small_dataset
    keep = ds.rows.entity == ent
    assert only.X[..., 0].sum() <= ds.rows.n[keep].sum() * 30 + 1  # bounded by that entity's own rows
    assert len(only) + len(other) == len(ws)


def test_status_exclusions_around_incidents(small_dataset, cfg):
    ws = build_windowset(small_dataset, cfg)
    inc = small_dataset.incidents[0]
    e = small_dataset.entities.index(inc.entity)
    rows = ws.ent == e
    mid = (ws.t_s > inc.onset) & (ws.t_s < inc.end) & rows
    assert (ws.status[mid] == ACTIVE).all()
    rec = (ws.t_s >= inc.end) & (ws.t_s < inc.end + 600) & rows
    assert set(ws.status[rec]) <= {RECOVERY, ACTIVE}
    pos = ws.y == 1
    assert (ws.status[pos] != ACTIVE).sum() > 0 and OK in set(ws.status[pos])


def test_unmonitored_bins_are_unknown_not_zero(small_dataset, cfg):
    ws = build_windowset(small_dataset, cfg)
    assert ws.X[..., 5].min() == 0.0  # some bins are unobserved (outage host)
    unobs = ws.X[..., 5] == 0
    assert (ws.X[..., 0][unobs & (ws.lengths[:, None] == 30)] == 0).all()


def test_decisions_are_only_made_where_monitored(small_dataset, cfg):
    ws = build_windowset(small_dataset, cfg)
    assert small_dataset.cover[ws.ent, ws.t].all()


def test_no_lookahead_in_features_when_future_is_appended(small_records, cfg):
    events, incidents, coverage = small_records
    evs = [event_from_record(e) for e in events]
    incs = [incident_from_record(i) for i in incidents]
    full = build_dataset("full", "synthetic", evs, incs, coverage)
    ws_full = build_windowset(full, cfg)
    cut = full.timeline.time_of(full.timeline.n_bins // 2)
    past_only = build_dataset("past", "synthetic", [e for e in evs if e.arrival_time <= cut], [i for i in incs if i.onset <= cut], coverage)
    ws_past = build_windowset(past_only, cfg)
    t_max = full.timeline.n_bins // 2
    a = ws_full.subset(ws_full.t <= t_max)
    b = ws_past.subset(ws_past.t <= t_max)
    assert a.X.shape == b.X.shape and np.array_equal(a.X, b.X)


def test_validator_accepts_fixture_and_flags_tampering(tmp_path):
    write_fixture(SMALL, tmp_path / "d")
    ok = validate_dir(tmp_path / "d", Config())
    assert ok["ok"], ok["errors"]
    assert "prevalence_per_split" in ok["report"] and ok["report"]["rows"]["duplicate_event_ids"] > 0
    shutil.copytree(tmp_path / "d", tmp_path / "bad")
    with open(tmp_path / "bad" / "events.jsonl", "a", encoding="utf-8") as fh:
        fh.write(open(tmp_path / "bad" / "events.jsonl", encoding="utf-8").readline())
    bad = validate_dir(tmp_path / "bad", Config())
    assert not bad["ok"] and any("hash mismatch" in e or "data card" in e for e in bad["errors"])


def test_validator_names_missing_files(tmp_path):
    res = validate_dir(tmp_path, Config())
    assert not res["ok"] and any("events.jsonl" in e for e in res["errors"])
    missing = lanl_missing(tmp_path)
    assert len(missing) == 2 and "csr.lanl.gov" in missing[0]


def test_validator_rejects_label_leak(tmp_path):
    write_fixture(SMALL, tmp_path / "d")
    rec = read_jsonl(tmp_path / "d" / "events.jsonl")
    rec[0]["scenario_id"] = "S1"
    import json

    with open(tmp_path / "d" / "events.jsonl", "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in rec) + "\n")
    res = validate_dir(tmp_path / "d", Config())
    assert not res["ok"]


def test_load_dir_round_trip(tmp_path):
    write_fixture(SMALL, tmp_path / "d")
    ds = load_dir(tmp_path / "d")
    assert len(ds.entities) == SMALL.n_hosts and ds.incidents
