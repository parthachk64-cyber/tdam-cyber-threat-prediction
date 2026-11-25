"""Typed records. Ground truth (incidents) lives apart from telemetry (events)."""

from __future__ import annotations

from dataclasses import dataclass

EVENT_FIELDS = ("event_id", "entity", "event_time", "arrival_time", "modality", "outcome", "dst")
INCIDENT_FIELDS = ("incident_id", "entity", "onset", "end", "label_available", "kind")
# Field names that must never appear in telemetry (they would leak the target).
FORBIDDEN_EVENT_FIELDS = frozenset(
    {"label", "incident", "incident_id", "scenario", "scenario_id", "verdict", "is_attack", "onset", "y"}
)
OUTCOMES = ("success", "fail")


@dataclass(frozen=True)
class Event:
    event_id: str
    entity: str
    event_time: int  # epoch seconds: when it happened
    arrival_time: int  # epoch seconds: when a defender could first see it
    dst: str
    outcome: str  # "success" | "fail"
    modality: str = "auth"


@dataclass(frozen=True)
class Incident:
    """Onset, end, label-availability are kept as three separate times."""

    incident_id: str
    entity: str
    onset: int
    end: int
    label_available: int
    kind: str = "annotated_unusual_sequence"
