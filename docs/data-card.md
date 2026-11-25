# Data card: synthetic fixture

* **Generator:** `tdam.data.fixture` (original, seeded, numpy). Writes saved records only; no network, no scanning, no attack traffic.
* **Files:** `events.jsonl` (telemetry, no labels), `incidents.jsonl` (ground truth, separate), `coverage.json` (monitored intervals, absent modalities), `data_card.json` (counts, SHA-256).
* **Unit of analysis:** (entity, decision minute). **Label source:** `incidents.jsonl`.
* **Contents:** diurnal authentication volume; benign bursts; benign failure bursts; fan-out *look-alikes* with no incident; annotated incidents (75 % with a precursor ramp, 25 % abrupt); delayed records (15 %, some minutes late); duplicate deliveries (1.5 %); coverage outages on two hosts.
* **Default demo:** 16 hosts, 8 days, seed 7. `examples/sample/` holds a 3-host, 12-hour sample (1,171 events, 7 incidents).
* **Caveat:** the signal is built by the generator. Doing well here shows the pipeline works, not that anything is detected on a real network.
* **Validate:** `tdam data verify data/demo` checks schema, arrival >= event time, label-like fields, row counts, hashes, coverage, positives per split, duplicates and censoring.
* **Original study data:** *not identified* (the abstract names no dataset). Marked unverified; no familiar benchmark is assumed.
