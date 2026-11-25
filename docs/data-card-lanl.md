# Data card: LANL "Comprehensive, Multi-Source Cyber-Security Events"

Provider: Los Alamos National Laboratory, <https://csr.lanl.gov/data/cyber1/>. Cite A. D. Kent, "Cybersecurity Data Sources for Dynamic Network Research," *Dynamic Networks in Cybersecurity*, 2015, and the dataset record (doi:10.11578/1366450). Check the provider page for current terms; the owner downloads the files, which stay out of Git.

* **Expected files:** `auth.txt.gz` (time, src user, dst user, src computer, dst computer, auth type, logon type, orientation, success/failure) and `redteam.txt.gz` (time, user, src computer, dst computer). The SHA-256 of each is written to the prepared slice's `data_card.json`.
* **Unit:** source computer x minute. **Time:** seconds elapsed since collection start, rendered on an arbitrary 1970 epoch (not a calendar date).
* **Labels:** red-team lists *specific compromise events*, not incident boundaries. Episodes are formed per source computer by a deterministic gap rule (default 60 min); **onset = first event, a proxy**.
* **Arrival time:** not provided. Default replay assumption: arrival = event time. `--simulated-delay-max-s` adds a separately labelled random delay.
* **Coverage:** a computer is assumed monitored from its first observed bin to the end of the slice; silent bins are true zeros (recorded in `coverage.json`).
* **Modalities:** authentication only. DNS, flow and process are not invented.
* **Unmatched warnings** are "unmatched to known labels", not proven false alarms: red-team coverage is incomplete.
* **Commands:**
  ```
  tdam data verify --lanl /path/to/downloads     # prints exactly what is missing; never falls back to synthetic
  tdam data prepare-lanl --auth auth.txt.gz --redteam redteam.txt.gz --out data/lanl --start-day 0 --max-days 7 --top-k 100
  tdam run --lanl-dir data/lanl --out runs/lanl --report docs/results_lanl.md
  ```
  Use a bounded slice (`--max-days`, `--top-k`, and `decision_stride` in the config); the full 58-day file does not fit densely in memory.
* **Status in this repository:** adapter implemented and tested on a tiny file in LANL's column layout (format test only). **No LANL run has been executed.**
