"""Seeded, original synthetic telemetry generator.

Writes saved records only. It never touches a network and generates no attack
traffic: "incidents" are annotated *patterns in synthetic authentication logs*.
Ground truth goes to a separate file; the event stream carries no labels.

Pattern vocabulary (all synthetic):
  normal       diurnal Poisson authentication volume, habitual destinations
  benign burst volume spike, normal failure rate, habitual destinations
  benign fail  failure-rate spike on habitual destinations (mistyped passwords)
  look-alike   destination fan-out ramp that is NOT followed by an incident
  incident     optional precursor ramp (fan-out + failures), then an annotated
               high fan-out phase whose first minute is the onset
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tdam.data.io import sha256_file, write_json, write_jsonl
from tdam.timeutil import iso, parse_iso

POOL = 240
HABITUAL = 8


@dataclass(frozen=True)
class FixtureSpec:
    n_hosts: int = 16
    days: float = 8.0
    seed: int = 7
    start: str = "2026-01-05T00:00:00Z"
    incidents_per_host_day: float = 0.7
    precursor_probability: float = 0.75
    benign_bursts_per_host_day: float = 3.0
    benign_fail_per_host_day: float = 1.0
    lookalikes_per_host_day: float = 0.6
    outage_hosts: int = 2
    duplicate_rate: float = 0.015


def _diurnal(minute_of_day: np.ndarray) -> np.ndarray:
    hours = minute_of_day / 60.0
    return 0.35 + 0.65 * np.clip(np.sin((hours - 6.0) / 12.0 * np.pi), 0.0, None)


def _spaced_onsets(rng, n, lo, hi, gap, blocked):
    onsets: list[int] = []
    tries = 0
    while len(onsets) < n and tries < 400:
        tries += 1
        cand = int(rng.integers(lo, hi))
        if any(abs(cand - o) < gap for o in onsets):
            continue
        if any(a - 40 <= cand <= b + 40 for a, b in blocked):
            continue
        onsets.append(cand)
    return sorted(onsets)


def generate(spec: FixtureSpec):
    """Return (events, incidents, coverage) as plain dict records."""
    rng = np.random.default_rng(spec.seed)
    t0 = parse_iso(spec.start)
    n_min = int(round(spec.days * 1440))
    hosts = [f"H{i + 1:03d}" for i in range(spec.n_hosts)]
    outage_hosts = set(rng.choice(hosts, size=min(spec.outage_hosts, len(hosts)), replace=False)) if spec.outage_hosts else set()

    events: list[dict] = []
    incidents: list[dict] = []
    coverage: dict[str, list[list[str]]] = {}
    minute_of_day = np.arange(n_min) % 1440
    diurnal = _diurnal(minute_of_day)

    for h in hosts:
        base = float(rng.uniform(0.6, 1.4))
        mult = diurnal * base
        pfail = np.full(n_min, 0.03)
        pnew = np.full(n_min, 0.02)

        blocked: list[tuple[int, int]] = []
        monitored = [(0, n_min)]
        if h in outage_hosts:
            length = int(rng.integers(180, 360))
            start = int(rng.integers(1440, max(1441, n_min - length - 1440)))
            blocked.append((start, start + length))
            monitored = [(0, start), (start + length, n_min)]
        coverage[h] = [[iso(t0 + a * 60), iso(t0 + b * 60)] for a, b in monitored]

        def count(per_day):
            return int(rng.poisson(per_day * spec.days))

        for _ in range(count(spec.benign_bursts_per_host_day)):
            s, d = int(rng.integers(5, n_min - 20)), int(rng.integers(3, 9))
            mult[s : s + d] = mult[s : s + d] * rng.uniform(3.0, 5.0)
        for _ in range(count(spec.benign_fail_per_host_day)):
            s, d = int(rng.integers(5, n_min - 20)), int(rng.integers(3, 7))
            pfail[s : s + d] = 0.5
        for _ in range(count(spec.lookalikes_per_host_day)):
            s, d = int(rng.integers(30, n_min - 30)), int(rng.integers(8, 16))
            ramp = np.linspace(0.02, 0.5, d)
            pnew[s : s + d] = np.maximum(pnew[s : s + d], ramp)
            mult[s : s + d] = mult[s : s + d] * 1.5

        onsets = _spaced_onsets(
            rng, count(spec.incidents_per_host_day), 120, n_min - 120, gap=120, blocked=blocked
        )
        for k, onset in enumerate(onsets):
            if rng.random() < spec.precursor_probability:
                length = int(rng.integers(8, 21))
                lo = max(0, onset - length)
                n = onset - lo
                pnew[lo:onset] = np.maximum(pnew[lo:onset], np.linspace(0.05, 0.6, n))
                pfail[lo:onset] = np.maximum(pfail[lo:onset], np.linspace(0.03, 0.4, n))
                mult[lo:onset] = mult[lo:onset] * np.linspace(1.0, 2.5, n)
            dur = int(rng.integers(6, 15))
            end = min(n_min - 1, onset + dur)
            pnew[onset:end] = 0.9
            pfail[onset:end] = 0.6
            mult[onset:end] = np.maximum(mult[onset:end], 5.0 * base)
            delay = int(rng.integers(30, 121))
            incidents.append(
                {
                    "incident_id": f"{h}-I{k + 1:02d}",
                    "entity": h,
                    "onset": iso(t0 + onset * 60),
                    "end": iso(t0 + end * 60),
                    "label_available": iso(t0 + (end + delay) * 60),
                    "kind": "annotated_unusual_sequence",
                }
            )

        habitual = rng.choice(POOL, size=HABITUAL, replace=False)
        counts = rng.poisson(mult)
        for a, b in blocked:
            counts[a:b] = 0
        for m in np.nonzero(counts)[0]:
            n = int(counts[m])
            secs = rng.integers(0, 60, size=n)
            fails = rng.random(n) < pfail[m]
            novel = rng.random(n) < pnew[m]
            dsts = np.where(novel, rng.integers(0, POOL, size=n), rng.choice(habitual, size=n))
            r = rng.random(n)
            delays = np.where(
                r < 0.85,
                rng.integers(0, 11, size=n),
                np.where(r < 0.97, rng.integers(60, 181, size=n), rng.integers(300, 1201, size=n)),
            )
            for j in range(n):
                ev_s = t0 + int(m) * 60 + int(secs[j])
                events.append(
                    {
                        "entity": h,
                        "event_time": ev_s,
                        "arrival_time": ev_s + int(delays[j]),
                        "outcome": "fail" if fails[j] else "success",
                        "dst": f"N{int(dsts[j]):04d}",
                        "modality": "auth",
                    }
                )

    events.sort(key=lambda e: (e["event_time"], e["entity"], e["dst"]))
    for i, e in enumerate(events):
        e["event_id"] = f"e{i:07d}"
    dup_idx = rng.random(len(events)) < spec.duplicate_rate
    dups = []
    for i in np.nonzero(dup_idx)[0]:
        d = dict(events[int(i)])
        d["arrival_time"] = d["arrival_time"] + int(rng.integers(60, 301))
        dups.append(d)
    events.extend(dups)
    events.sort(key=lambda e: (e["arrival_time"], e["event_id"]))
    for e in events:
        e["event_time"] = iso(e["event_time"])
        e["arrival_time"] = iso(e["arrival_time"])
    incidents.sort(key=lambda r: (r["onset"], r["entity"]))

    coverage_doc = {
        "entities": coverage,
        "modalities_present": ["auth"],
        "modalities_absent": ["dns", "flow", "process"],
        "note": "Intervals are monitored time; outside them counts are UNKNOWN, not zero.",
    }
    return events, incidents, coverage_doc


def write_fixture(spec: FixtureSpec, out_dir: str | Path) -> dict:
    """Generate and write events.jsonl, incidents.jsonl, coverage.json, data_card.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events, incidents, coverage = generate(spec)
    write_jsonl(out / "events.jsonl", events)
    write_jsonl(out / "incidents.jsonl", incidents)
    write_json(out / "coverage.json", coverage)
    card = {
        "name": "tdam-synthetic-fixture",
        "kind": "synthetic (original generator, seeded)",
        "spec": asdict(spec),
        "n_events": len(events),
        "n_incidents": len(incidents),
        "n_entities": spec.n_hosts,
        "files": {
            name: {"sha256": sha256_file(out / name)}
            for name in ("events.jsonl", "incidents.jsonl", "coverage.json")
        },
        "label_source": "incidents.jsonl, generated alongside events and never written into events.jsonl",
        "unit_of_analysis": "(entity, decision minute)",
        "licence": "MIT (original synthetic data)",
        "caveat": "Signals are constructed by this generator; results on it are demonstrations, not evidence about real networks.",
    }
    write_json(out / "data_card.json", card)
    return card
