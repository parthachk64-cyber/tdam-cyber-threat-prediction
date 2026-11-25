"""A loaded study dataset: bin rows + coverage + incidents on one timeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tdam.data.io import read_events, read_incidents, read_json
from tdam.data.schema import Event, Incident
from tdam.temporal.binning import BinRows, coverage_mask, events_to_rows
from tdam.timeutil import Timeline, parse_iso


@dataclass
class Dataset:
    name: str
    source: str  # "synthetic" | "independent-lanl" | "paper-original" | "user-supplied"
    timeline: Timeline
    entities: list[str]
    rows: BinRows
    cover: np.ndarray  # [E, n_bins] bool
    incidents: list[Incident]
    events: list[Event] | None = None
    meta: dict = field(default_factory=dict)


def build_dataset(
    name: str,
    source: str,
    events: list[Event],
    incidents: list[Incident],
    coverage: dict,
    bin_seconds: int = 60,
    meta: dict | None = None,
) -> Dataset:
    entities = sorted(coverage["entities"])
    starts = [parse_iso(lo) for e in entities for lo, _ in coverage["entities"][e]]
    stops = [parse_iso(hi) for e in entities for _, hi in coverage["entities"][e]]
    t0 = min(starts)
    n_bins = int((max(stops) - t0) // bin_seconds) + 1
    timeline = Timeline(t0=t0, n_bins=n_bins, bin_seconds=bin_seconds)
    rows, stats = events_to_rows(events, entities, timeline)
    cover = coverage_mask(coverage, entities, timeline)
    info = dict(meta or {})
    info["ingest_stats"] = stats
    return Dataset(name, source, timeline, entities, rows, cover, incidents, events, info)


def load_dir(path: str | Path, source: str = "synthetic", bin_seconds: int = 60) -> Dataset:
    p = Path(path)
    events = read_events(p / "events.jsonl")
    incidents = read_incidents(p / "incidents.jsonl")
    coverage = read_json(p / "coverage.json")
    card = read_json(p / "data_card.json") if (p / "data_card.json").exists() else {}
    return build_dataset(p.name, source, events, incidents, coverage, bin_seconds, {"data_card": card, "source_dir": str(p)})
