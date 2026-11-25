"""Dataset validator: schema, chronology, counts, hashes, coverage, prevalence per split, censoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tdam.config import Config
from tdam.data.dataset import build_dataset
from tdam.data.io import read_json, read_jsonl, sha256_file
from tdam.data.schema import FORBIDDEN_EVENT_FIELDS
from tdam.data.io import event_from_record, incident_from_record
from tdam.temporal.labels import CENSORED
from tdam.temporal.splits import make_partitions, partition_mask
from tdam.temporal.windows import build_windowset

REQUIRED = ("events.jsonl", "incidents.jsonl", "coverage.json")


def validate_dir(path: str | Path, cfg: Config | None = None) -> dict:
    """Return {"ok": bool, "errors": [...], "warnings": [...], "report": {...}}. Never silently passes."""
    cfg = cfg or Config()
    p = Path(path)
    errors: list[str] = []
    warns: list[str] = []
    report: dict = {}
    missing = [f for f in REQUIRED if not (p / f).exists()]
    if missing:
        return {
            "ok": False,
            "errors": [f"missing file: {p / f}" for f in missing],
            "warnings": [],
            "report": {},
            "how_to_fix": "generate the synthetic fixture with `tdam data generate --out <dir>`",
        }
    ev_rec = read_jsonl(p / "events.jsonl")
    inc_rec = read_jsonl(p / "incidents.jsonl")
    coverage = read_json(p / "coverage.json")

    try:
        events = [event_from_record(r) for r in ev_rec]
        incidents = [incident_from_record(r) for r in inc_rec]
    except (ValueError, KeyError) as exc:
        return {"ok": False, "errors": [f"schema: {exc}"], "warnings": [], "report": {}}
    leaked = {k for r in ev_rec for k in r} & FORBIDDEN_EVENT_FIELDS
    if leaked:
        errors.append(f"events carry label-like fields: {sorted(leaked)}")

    bad_order = sum(e.arrival_time < e.event_time for e in events)
    if bad_order:
        errors.append(f"{bad_order} events arrive before they happen")
    arrivals = np.array([e.arrival_time for e in events])
    if len(arrivals) and (np.diff(arrivals) < 0).any():
        warns.append("events.jsonl is not sorted by arrival_time (replay will sort it)")
    ids = [e.event_id for e in events]
    dup = len(ids) - len(set(ids))
    report["rows"] = {"events": len(events), "incidents": len(incidents), "duplicate_event_ids": dup}
    late = np.array([e.arrival_time - e.event_time for e in events]) if events else np.zeros(0)
    report["arrival_delay_s"] = {"max": int(late.max()) if len(late) else 0, "share_over_60s": float((late > 60).mean()) if len(late) else 0.0}

    for inc in incidents:
        if not (inc.onset < inc.end <= inc.label_available):
            errors.append(f"{inc.incident_id}: need onset < end <= label_available")
    ent_set = set(coverage["entities"])
    unknown = {e.entity for e in events} - ent_set
    if unknown:
        errors.append(f"events for entities with no coverage entry: {sorted(unknown)[:5]}")

    card_path = p / "data_card.json"
    if card_path.exists():
        card = read_json(card_path)
        if card.get("n_events") != len(events):
            errors.append(f"data card says {card.get('n_events')} events, file has {len(events)}")
        for name, info in card.get("files", {}).items():
            if (p / name).exists() and sha256_file(p / name) != info["sha256"]:
                errors.append(f"hash mismatch for {name}")
        report["data_card"] = {"checked": True}
    else:
        warns.append("no data_card.json: row counts and hashes were not cross-checked")

    if errors:
        return {"ok": False, "errors": errors, "warnings": warns, "report": report}

    ds = build_dataset(p.name, "synthetic", events, incidents, coverage, cfg.protocol.bin_seconds)
    report["coverage"] = {
        "start": ds.timeline.t0,
        "n_bins": ds.timeline.n_bins,
        "monitored_fraction": float(ds.cover.mean()),
        "entities": len(ds.entities),
    }
    ws = build_windowset(ds, cfg)
    w, h = cfg.protocol.lookback_bins, cfg.protocol.horizon_minutes * 60 // cfg.protocol.bin_seconds
    parts = make_partitions(ds.timeline.n_bins, {"train": cfg.splits.train, "val": cfg.splits.val, "calib": cfg.splits.calib, "test": cfg.splits.test})
    prev = {}
    for name, part in parts.items():
        m = partition_mask(ws.t, part, w, h) & ws.usable
        prev[name] = {"windows": int(m.sum()), "positives": int(ws.y[m].sum()), "prevalence": float(ws.y[m].mean()) if m.any() else None}
        if m.any() and ws.y[m].sum() == 0:
            warns.append(f"partition '{name}' has no positive windows; metrics there are undefined")
    report["prevalence_per_split"] = prev
    report["censored_windows"] = int((ws.status == CENSORED).sum())
    report["duplicate_event_ids_deduplicated"] = ds.meta["ingest_stats"]["duplicates"]
    return {"ok": True, "errors": [], "warnings": warns, "report": report}


def lanl_missing(directory: str | Path) -> list[str]:
    """Exact files and steps needed for the LANL benchmark. Never silently falls back to synthetic."""
    d = Path(directory)
    need = {
        "auth.txt.gz": "https://csr.lanl.gov/data/cyber1/auth.txt.gz",
        "redteam.txt.gz": "https://csr.lanl.gov/data/cyber1/redteam.txt.gz",
    }
    return [f"{d / n}  <- download from {u} (owner-accepted terms)" for n, u in need.items() if not (d / n).exists()]
