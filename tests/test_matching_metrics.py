import numpy as np

from tdam.data.schema import Incident
from tdam.evaluation.alerting import issue_alerts
from tdam.evaluation.matching import match_alerts
from tdam.evaluation.metrics import f1_optimal_threshold, lead_time_summary, window_metrics
from tdam.timeutil import parse_iso

T = lambda hm: parse_iso(f"2026-03-02T{hm}:00Z")  # noqa: E731


def inc(i, ent, onset, end, avail="23:00"):
    return Incident(i, ent, T(onset), T(end), T(avail))


def test_no_double_credit_and_redundant_alerts():
    a = inc("A", "H1", "10:20", "10:30")
    res = match_alerts([("H1", T("10:16")), ("H1", T("10:18"))], [a], 300)
    row = res.ledger[0]
    assert row.status == "early" and row.lead_min == 4  # earliest qualifying alert gets the credit
    assert res.redundant == 1 and not res.unmatched


def test_overlapping_incidents_prefer_unclaimed_upcoming_then_active():
    a, b = inc("A", "H1", "10:20", "10:30"), inc("B", "H1", "10:27", "10:40")
    res = match_alerts([("H1", T("10:23")), ("H1", T("10:25"))], [a, b], 300)
    status = {r.incident_id: r.status for r in res.ledger}
    assert status == {"A": "late", "B": "early"}


def test_alert_outside_horizon_is_unmatched_not_credited():
    a = inc("A", "H1", "10:20", "10:30")
    res = match_alerts([("H1", T("10:10")), ("H2", T("10:18"))], [a], 300)
    assert res.ledger[0].status == "missed" and len(res.unmatched) == 2


def test_no_alerts_gives_zero_early_and_all_missed():
    res = match_alerts([], [inc("A", "H1", "10:20", "10:30")], 300)
    c = res.counts()
    assert c["early"] == 0 and c["missed"] == 1 and c["unmatched_alerts"] == 0


def test_cooldown_groups_alerts_and_keeps_first_timestamp():
    ent = np.zeros(6, int)
    t = np.array([0, 60, 120, 600, 660, 1300])
    idx = issue_alerts(ent, t, np.ones(6), 0.5, 600)
    assert list(t[idx]) == [0, 600, 1300]


def test_single_class_metrics_are_undefined_not_zero():
    m = window_metrics(np.zeros(10), np.linspace(0, 1, 10), 0.5)
    assert m["average_precision"] is None and m["roc_auc"] is None
    assert m["recall"] is None  # no positives: recall is 0/0
    assert m["precision"] == 0.0  # alerts fired and none could be correct
    assert m["accuracy"] == 0.5
    assert m["always_negative_accuracy"] == 1.0


def test_confusion_counts_and_always_negative_baseline():
    y = np.array([0, 0, 0, 1, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.1])
    m = window_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 3)
    assert abs(m["always_negative_accuracy"] - 4 / 6) < 1e-12


def test_f1_threshold_without_positives_means_never_alert():
    thr = f1_optimal_threshold(np.zeros(5), np.array([0.2, 0.4, 0.1, 0.9, 0.3]))
    assert thr > 0.9


def test_lead_summary_empty():
    assert lead_time_summary([])["n"] == 0
