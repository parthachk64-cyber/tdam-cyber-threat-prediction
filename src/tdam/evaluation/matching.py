"""Alert -> incident matching with a per-incident ledger.

Rules (deterministic, no double credit):
  * Alerts are processed in (entity, time, alert order). An alert's time is the decision time at
    which it was issued and is never edited afterwards.
  * EARLY: the alert precedes the entity's next onset by at most H and that incident has no early
    credit yet. Lead = onset - alert time. Only the earliest qualifying alert earns credit.
  * LATE: otherwise, if an incident is in progress (onset <= t < end) and has no credit at all,
    the alert is credited as a detection after onset (delay = t - onset, lead credit zero).
  * REDUNDANT: the alert belongs to an incident that is already credited (repeat warning).
  * UNMATCHED: the alert maps to no annotated incident. This means "unmatched to known labels";
    annotations are incomplete, so unmatched is not proof of a false positive.
Overlapping incidents: an alert prefers an unclaimed *upcoming* incident within H, then an
unclaimed *active* one.
"""

from __future__ import annotations

from dataclasses import dataclass

from tdam.data.schema import Incident
from tdam.timeutil import iso


@dataclass
class LedgerRow:
    incident_id: str
    entity: str
    onset: int
    status: str  # early | late | missed
    alert_time: int | None = None
    lead_min: float = 0.0  # early-warning credit in minutes (zero unless early)
    delay_min: float | None = None  # detection delay after onset (late only)

    def as_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "entity": self.entity,
            "onset": iso(self.onset),
            "status": self.status,
            "first_credited_alert": iso(self.alert_time) if self.alert_time is not None else "",
            "lead_min": self.lead_min,
            "delay_min": "" if self.delay_min is None else self.delay_min,
        }


@dataclass
class MatchResult:
    ledger: list[LedgerRow]
    unmatched: list[tuple[str, int]]
    redundant: int
    n_alerts: int

    def counts(self) -> dict:
        early = sum(r.status == "early" for r in self.ledger)
        late = sum(r.status == "late" for r in self.ledger)
        missed = sum(r.status == "missed" for r in self.ledger)
        return {
            "incidents": len(self.ledger),
            "early": early,
            "late_only": late,
            "missed": missed,
            "unmatched_alerts": len(self.unmatched),
            "redundant_alerts": self.redundant,
            "alerts": self.n_alerts,
        }


def match_alerts(alerts: list[tuple[str, int]], incidents: list[Incident], horizon_s: int) -> MatchResult:
    by_ent: dict[str, list[Incident]] = {}
    for inc in sorted(incidents, key=lambda i: (i.onset, i.incident_id)):
        by_ent.setdefault(inc.entity, []).append(inc)
    ledger = {inc.incident_id: LedgerRow(inc.incident_id, inc.entity, inc.onset, "missed") for inc in incidents}
    unmatched: list[tuple[str, int]] = []
    redundant = 0
    for ent, t in sorted(alerts, key=lambda a: (a[0], a[1])):
        incs = by_ent.get(ent, [])
        upcoming = [i for i in incs if i.onset > t and i.onset - t <= horizon_s]
        active = [i for i in incs if i.onset <= t < i.end]
        credited = False
        for inc in upcoming:  # nearest onset first (sorted)
            row = ledger[inc.incident_id]
            if row.status != "early":
                row.status, row.alert_time, row.lead_min, row.delay_min = "early", t, (inc.onset - t) / 60.0, None
                credited = True
                break
        if not credited:
            for inc in active:
                row = ledger[inc.incident_id]
                if row.status == "missed":
                    row.status, row.alert_time, row.lead_min, row.delay_min = "late", t, 0.0, (t - inc.onset) / 60.0
                    credited = True
                    break
        if credited:
            continue
        if upcoming or active:
            redundant += 1
        else:
            unmatched.append((ent, t))
    return MatchResult(list(ledger.values()), unmatched, redundant, len(alerts))
