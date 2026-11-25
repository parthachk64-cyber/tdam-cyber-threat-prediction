"""Regression tests for the three worked examples in the study protocol (10:20 UTC incident)."""
import numpy as np

from tdam.data.schema import Event, Incident
from tdam.evaluation.matching import match_alerts
from tdam.temporal.binning import build_entity_windows, events_to_rows
from tdam.temporal.labels import ACTIVE, OK, RECOVERY, label_windows
from tdam.timeutil import Timeline, parse_iso

T0 = parse_iso("2026-03-02T10:00:00Z")
TL = Timeline(T0, 60)
ENT = ["H1"]
W = 30


def ev(i, hhmmss, arrive, dst="N1", outcome="success"):
    base = "2026-03-02T"
    return Event(f"e{i}", "H1", parse_iso(base + hhmmss + "Z"), parse_iso(base + arrive + "Z"), dst, outcome)


def window_at(events, hhmm):
    rows, _ = events_to_rows(events, ENT, TL)
    t = int(TL.bin_of(parse_iso(f"2026-03-02T{hhmm}:00Z")))
    cover = np.ones(TL.n_bins, dtype=bool)
    X, _ = build_entity_windows(rows, 0, TL, W, cover, np.array([t]))
    return X[0]


def test_record_stamped_1016_ingested_1018_invisible_to_1017():
    late = ev(1, "10:16:00", "10:18:00")
    assert window_at([late], "10:17")[:, 0].sum() == 0
    assert window_at([late], "10:18")[:, 0].sum() == 1


def test_record_visible_when_arrived_in_time():
    prompt = ev(1, "10:16:00", "10:16:05")
    assert window_at([prompt], "10:17")[:, 0].sum() == 1


def test_event_after_decision_time_never_included():
    future = ev(1, "10:17:30", "10:17:31")
    assert window_at([future], "10:17")[:, 0].sum() == 0


def test_history_endpoints_half_open():
    # event exactly W minutes before t is outside (t-W, t]; W-1 minutes before is inside
    out = ev(1, "10:20:00", "10:20:00")
    inside = ev(2, "10:21:00", "10:21:00")
    X = window_at([out, inside], "10:50")
    assert X[:, 0].sum() == 1


def test_duplicate_ids_counted_once():
    a = ev(1, "10:16:00", "10:16:05")
    b = ev(1, "10:16:00", "10:19:00")
    rows, stats = events_to_rows([a, b], ENT, TL)
    assert stats["duplicates"] == 1 and rows.n.sum() == 1


def test_unique_destinations_respect_arrival():
    a = ev(1, "10:16:00", "10:16:05", dst="A")
    b = ev(2, "10:16:10", "10:18:00", dst="B")
    assert window_at([a, b], "10:17")[:, 3].sum() == 1
    assert window_at([a, b], "10:18")[:, 3].sum() == 2


def test_label_horizon_endpoints_and_status():
    inc = Incident("I1", "H1", parse_iso("2026-03-02T10:20:00Z"), parse_iso("2026-03-02T10:30:00Z"),
                   parse_iso("2026-03-02T11:00:00Z"))
    times = np.array([parse_iso(f"2026-03-02T{h}:00Z") for h in ("10:14", "10:15", "10:19", "10:20", "10:25", "10:35", "10:41")])
    y, status, known = label_windows(np.zeros(len(times), int), times, [inc], ENT, 300, 600, parse_iso("2026-03-02T12:00:00Z"))
    assert list(y) == [0, 1, 1, 0, 0, 0, 0]  # (t, t+5]: 10:15 positive, 10:20 itself is not (onset not > t)
    assert status[3] == ACTIVE and status[4] == ACTIVE
    assert status[5] == RECOVERY and status[6] == OK
    assert known[1] == parse_iso("2026-03-02T11:00:00Z")  # label settles only when the annotation is delivered


def test_early_late_examples_from_protocol():
    inc = Incident("I1", "H1", parse_iso("2026-03-02T10:20:00Z"), parse_iso("2026-03-02T10:30:00Z"),
                   parse_iso("2026-03-02T11:00:00Z"))
    early = match_alerts([("H1", parse_iso("2026-03-02T10:17:00Z"))], [inc], 300)
    assert early.ledger[0].status == "early" and early.ledger[0].lead_min == 3
    late = match_alerts([("H1", parse_iso("2026-03-02T10:21:00Z"))], [inc], 300)
    row = late.ledger[0]
    assert row.status == "late" and row.delay_min == 1 and row.lead_min == 0
