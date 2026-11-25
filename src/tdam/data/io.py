"""JSONL/CSV readers and writers for events, incidents and coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from tdam.data.schema import EVENT_FIELDS, FORBIDDEN_EVENT_FIELDS, Event, Incident
from tdam.timeutil import iso, parse_iso


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
    return out


def event_from_record(rec: dict) -> Event:
    leaked = FORBIDDEN_EVENT_FIELDS & set(rec)
    if leaked:
        raise ValueError(f"event record carries label-like fields {sorted(leaked)}")
    missing = [f for f in ("event_id", "entity", "event_time", "arrival_time", "dst", "outcome") if f not in rec]
    if missing:
        raise ValueError(f"event record missing fields {missing}")
    if rec["outcome"] not in ("success", "fail"):
        raise ValueError(f"unknown outcome {rec['outcome']!r}")
    return Event(
        event_id=str(rec["event_id"]),
        entity=str(rec["entity"]),
        event_time=parse_iso(rec["event_time"]),
        arrival_time=parse_iso(rec["arrival_time"]),
        dst=str(rec["dst"]),
        outcome=rec["outcome"],
        modality=str(rec.get("modality", "auth")),
    )


def event_to_record(ev: Event) -> dict:
    return {
        "event_id": ev.event_id,
        "entity": ev.entity,
        "event_time": iso(ev.event_time),
        "arrival_time": iso(ev.arrival_time),
        "modality": ev.modality,
        "outcome": ev.outcome,
        "dst": ev.dst,
    }


def read_events(path: str | Path) -> list[Event]:
    """Read events from .jsonl or .csv (columns as in EVENT_FIELDS)."""
    path = Path(path)
    if path.suffix == ".csv":
        frame = pd.read_csv(path, dtype=str)
        records = frame.to_dict("records")
    else:
        records = read_jsonl(path)
    return [event_from_record(r) for r in records]


def incident_from_record(rec: dict) -> Incident:
    return Incident(
        incident_id=str(rec["incident_id"]),
        entity=str(rec["entity"]),
        onset=parse_iso(rec["onset"]),
        end=parse_iso(rec["end"]),
        label_available=parse_iso(rec["label_available"]),
        kind=str(rec.get("kind", "annotated_unusual_sequence")),
    )


def incident_to_record(inc: Incident) -> dict:
    return {
        "incident_id": inc.incident_id,
        "entity": inc.entity,
        "onset": iso(inc.onset),
        "end": iso(inc.end),
        "label_available": iso(inc.label_available),
        "kind": inc.kind,
    }


def read_incidents(path: str | Path) -> list[Incident]:
    return [incident_from_record(r) for r in read_jsonl(path)]


def write_json(path: str | Path, obj) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


__all__ = [
    "EVENT_FIELDS",
    "event_from_record",
    "event_to_record",
    "incident_from_record",
    "incident_to_record",
    "read_events",
    "read_incidents",
    "read_json",
    "read_jsonl",
    "sha256_file",
    "write_json",
    "write_jsonl",
]
