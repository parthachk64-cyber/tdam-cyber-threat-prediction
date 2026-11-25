"""Streaming adapter for LANL "Comprehensive, Multi-Source Cyber-Security Events".

Provider: Los Alamos National Laboratory, https://csr.lanl.gov/data/cyber1/ (owner downloads the
files; they stay outside Git). Cite: A. D. Kent, "Cybersecurity Data Sources for Dynamic Network
Research," in Dynamic Networks in Cybersecurity, 2015; and A. D. Kent, "Comprehensive, Multi-Source
Cyber-Security Events," LANL, 2015, doi:10.11578/1366450.

Files used: auth.txt.gz  (time,src user,dst user,src computer,dst computer,auth type,logon type,orientation,result)
            redteam.txt.gz (time,user,src computer,dst computer)
`time` is seconds elapsed since the start of collection; the calendar date is not given, so times are
rendered on an arbitrary 1970 epoch ("elapsed time", not a real date).

Honest limits (also in docs/data-card-lanl.md)
  * redteam.txt lists specific compromise events, NOT incident boundaries. Episodes are defined by a
    deterministic gap rule over each source computer and the onset is a PROXY (first red-team event).
  * Alerts that match no red-team episode are "unmatched to known labels", not proven false alarms.
  * No ingestion timestamp exists. Default: arrival = event time (replay assumption). A separately
    labelled simulated delay can be applied.
  * Only the authentication modality is used; DNS/flow/process are not invented.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tdam.data.dataset import Dataset
from tdam.data.io import sha256_file, write_json, write_jsonl
from tdam.data.schema import Incident
from tdam.temporal.binning import BinRows, coverage_mask
from tdam.timeutil import Timeline, ceil_div, iso

AUTH_USECOLS = [0, 3, 4, 8]
AUTH_NAMES = ["time", "src", "dst", "outcome"]
STATE = "state.json"


def _read_auth(path, skip: int, chunksize: int):
    return pd.read_csv(path, header=None, usecols=AUTH_USECOLS, names=AUTH_NAMES, skiprows=skip, chunksize=chunksize, dtype={"time": np.int64, "src": str, "dst": str, "outcome": str}, compression="infer")


def count_sources(auth_path, start_s: int, end_s: int, chunksize: int = 2_000_000) -> Counter:
    """First pass: events per source computer inside [start_s, end_s). Stops once the sorted file passes end_s."""
    counts: Counter = Counter()
    for chunk in _read_auth(auth_path, 0, chunksize):
        sel = chunk[(chunk["time"] >= start_s) & (chunk["time"] < end_s)]
        counts.update(sel["src"].value_counts().to_dict())
        if chunk["time"].iloc[-1] >= end_s:
            break
    return counts


def read_redteam(path, start_s: int, end_s: int) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=["time", "user", "src", "dst"], compression="infer")
    return frame[(frame["time"] >= start_s) & (frame["time"] < end_s)].sort_values("time", kind="stable")


def episodes_from_redteam(red: pd.DataFrame, gap_s: int, label_delay_s: int, end_s: int) -> list[dict]:
    """Deterministic episodes: consecutive red-team events from one source within gap_s form an episode.
    Onset is the first event (a proxy); end is the last event + 1 s; label_available = end + delay."""
    out = []
    for src, grp in red.groupby("src", sort=True):
        times = grp["time"].to_numpy()
        start = times[0]
        prev = times[0]
        k = 0
        for t in list(times[1:]) + [None]:
            if t is None or t - prev > gap_s:
                k += 1
                end = int(prev) + 1
                out.append(
                    {
                        "incident_id": f"{src}-E{k:03d}",
                        "entity": src,
                        "onset": iso(int(start)),
                        "end": iso(end),
                        "label_available": iso(end + label_delay_s),
                        "kind": "lanl_redteam_episode_proxy_onset",
                    }
                )
                if t is not None:
                    start = t
            if t is not None:
                prev = t
    return out


def preprocess_auth(
    auth_path,
    out_dir,
    entities: list[str],
    start_s: int,
    end_s: int,
    bin_seconds: int = 60,
    chunksize: int = 2_000_000,
    arrival_delay_s: int = 0,
    simulated_delay_max_s: int | None = None,
    seed: int = 0,
    resume: bool = True,
) -> Path:
    """Chunked, resumable aggregation to bin rows. State (rows consumed + the unfinished last bin) is
    persisted after every chunk, so an interrupted run continues where it stopped."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / STATE
    state = {"rows_done": 0, "parts": 0, "done": False}
    carry = pd.DataFrame(columns=AUTH_NAMES)
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        if (out / "carry.csv").exists():
            carry = pd.read_csv(out / "carry.csv", dtype={"time": np.int64, "src": str, "dst": str, "outcome": str})
    elif not resume:
        for f in out.glob("part-*.csv.gz"):
            f.unlink()
    if state["done"]:
        return out
    allow = set(entities)
    rng = np.random.default_rng(seed)

    def flush(df: pd.DataFrame, index: int) -> None:
        if df.empty:
            return
        b = ceil_div(df["time"].to_numpy(), bin_seconds)
        df = df.assign(b=b, fail=(df["outcome"] == "Fail").astype(float))
        g = df.groupby(["src", "b"], sort=True).agg(n=("fail", "size"), nf=("fail", "sum"), nd=("dst", "nunique")).reset_index()
        delay = np.full(len(g), arrival_delay_s, dtype=np.int64)
        if simulated_delay_max_s:
            delay = delay + rng.integers(0, simulated_delay_max_s + 1, size=len(g))
        g["avail"] = g["b"] + ceil_div(delay, bin_seconds)
        g = g.rename(columns={"src": "entity"})[["entity", "b", "avail", "n", "nf", "nd"]]
        g.to_csv(out / f"part-{index:05d}.csv.gz", index=False)

    finished = True
    for chunk in _read_auth(auth_path, state["rows_done"], chunksize):
        consumed = len(chunk)
        sel = chunk[(chunk["time"] >= start_s) & (chunk["time"] < end_s) & chunk["src"].isin(allow)]
        past_end = bool(chunk["time"].iloc[-1] >= end_s)
        df = pd.concat([carry, sel], ignore_index=True)
        if df.empty:
            carry = df
        else:
            last_bin = int(ceil_div(df["time"].max(), bin_seconds))
            full = ceil_div(df["time"].to_numpy(), bin_seconds) < last_bin
            if past_end:
                flush(df, state["parts"])
                carry = df.iloc[0:0]
            else:
                flush(df[full], state["parts"])
                carry = df[~full]
            state["parts"] += 1
        state["rows_done"] += consumed
        carry.to_csv(out / "carry.csv", index=False)
        state_path.write_text(json.dumps(state))
        if past_end:
            break
    else:
        finished = True
    if not carry.empty:
        flush(carry, state["parts"])
        state["parts"] += 1
        carry = carry.iloc[0:0]
        carry.to_csv(out / "carry.csv", index=False)
    state["done"] = finished
    state_path.write_text(json.dumps(state))
    return out


def prepare_lanl(
    auth_path,
    redteam_path,
    out_dir,
    start_day: float = 0.0,
    max_days: float = 7.0,
    top_k: int = 100,
    gap_minutes: int = 60,
    label_delay_minutes: int = 0,
    arrival_delay_s: int = 0,
    simulated_delay_max_s: int | None = None,
    seed: int = 0,
    bin_seconds: int = 60,
    chunksize: int = 2_000_000,
    log=print,
) -> dict:
    """Prepare a bounded slice of LANL. Entity set = top-k busiest sources plus every red-team source."""
    out = Path(out_dir)
    start_s, end_s = int(start_day * 86400), int((start_day + max_days) * 86400)
    red = read_redteam(redteam_path, start_s, end_s)
    log(f"red-team events in window: {len(red)}")
    counts = count_sources(auth_path, start_s, end_s, chunksize)
    entities = sorted({e for e, _ in counts.most_common(top_k)} | set(red["src"]))
    log(f"entities: {len(entities)}")
    preprocess_auth(auth_path, out, entities, start_s, end_s, bin_seconds, chunksize, arrival_delay_s, simulated_delay_max_s, seed)
    parts = sorted(out.glob("part-*.csv.gz"))
    rows = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    rows = rows.groupby(["entity", "b", "avail"], sort=True).sum().reset_index()
    rows["b"] -= start_s // bin_seconds
    rows["avail"] -= start_s // bin_seconds
    rows.to_csv(out / "rows.csv.gz", index=False)
    episodes = episodes_from_redteam(red.assign(time=red["time"] - start_s), gap_minutes * 60, label_delay_minutes * 60, end_s - start_s)
    write_jsonl(out / "incidents.jsonl", episodes)
    n_bins = int((end_s - start_s) // bin_seconds) + 1
    first = rows.groupby("entity")["b"].min().to_dict()
    coverage = {
        "entities": {e: [[iso(first[e] * bin_seconds), iso((n_bins - 1) * bin_seconds)]] for e in entities if e in first},
        "modalities_present": ["auth"],
        "modalities_absent": ["dns", "flow", "process"],
        "assumption": "an entity is treated as monitored from its first observed bin to the end of the slice; silent bins are true zeros",
    }
    write_json(out / "coverage.json", coverage)
    card = {
        "name": "lanl-cyber1-slice",
        "source": "https://csr.lanl.gov/data/cyber1/",
        "inputs": {Path(auth_path).name: sha256_file(auth_path), Path(redteam_path).name: sha256_file(redteam_path)},
        "slice": {"start_s": start_s, "end_s": end_s, "top_k": top_k, "entities": len(entities), "elapsed_time_note": "seconds since collection start"},
        "label_rule": {"gap_minutes": gap_minutes, "onset": "first red-team event of the episode (PROXY)", "label_delay_minutes": label_delay_minutes},
        "arrival_model": {"arrival_delay_s": arrival_delay_s, "simulated_delay_max_s": simulated_delay_max_s, "seed": seed, "assumption": "no ingestion timestamps exist in LANL"},
        "episodes": len(episodes),
        "redteam_sources_missing_from_rows": sorted(set(red["src"]) - set(first)),
    }
    write_json(out / "data_card.json", card)
    return card


def load_lanl_dir(path, bin_seconds: int = 60) -> Dataset:
    from tdam.data.io import read_incidents, read_json

    p = Path(path)
    coverage = read_json(p / "coverage.json")
    entities = sorted(coverage["entities"])
    frame = pd.read_csv(p / "rows.csv.gz")
    n_bins = int(max(int(frame["b"].max()), int(frame["avail"].max()))) + 1
    from tdam.timeutil import parse_iso

    t_end = max(parse_iso(hi) for e in entities for _, hi in coverage["entities"][e])
    n_bins = max(n_bins, int(t_end // bin_seconds) + 1)
    timeline = Timeline(0, n_bins, bin_seconds)
    rows = BinRows.from_frame(frame, entities)
    cover = coverage_mask(coverage, entities, timeline)
    incidents: list[Incident] = [i for i in read_incidents(p / "incidents.jsonl") if i.entity in set(entities)]
    card = read_json(p / "data_card.json")
    return Dataset("lanl-slice", "independent-lanl", timeline, entities, rows, cover, incidents, None, {"data_card": card, "source_dir": str(p)})


def write_sample_in_lanl_format(path, n_minutes: int = 30) -> None:
    """Tiny file in LANL's column layout for FORMAT tests only. It is not LANL data."""
    lines = []
    for m in range(n_minutes):
        lines.append(f"{m * 60 + 1},U1@D,U1@D,C1,C{m % 3 + 2},Kerberos,Network,LogOn,Success")
        if m % 5 == 0:
            lines.append(f"{m * 60 + 2},U2@D,U2@D,C2,C1,NTLM,Network,LogOn,Fail")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
