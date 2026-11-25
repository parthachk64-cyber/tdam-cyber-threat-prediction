# Paper map

**Paper:** *Cyber Threat Prediction Using Temporal Data Analysis Model*, Partha Chakraborty, 2026 6th International Conference on Intelligent Technologies (CONIT), Belgavi, India, 19-21 June 2026.
DOI <https://doi.org/10.1109/CONIT69683.2026.11620680> · IEEE record <https://ieeexplore.ieee.org/document/11620680>. The publisher PDF is linked, not republished.

## What could and could not be inspected

| Item | Status |
|---|---|
| Abstract | Read (supplied with the task). |
| Full text, Sections III-V, equations, tables, figures | **Not read.** `ieeexplore.ieee.org` and `doi.org` answered the automated request with HTTP 202 (bot challenge) and no PDF was supplied. |
| Author-linked data, code, trained models, split definitions | **None found** in the supplied material or on the author's public GitHub account at the time of the check (repos: attention-qelm-gwo-reproduction, parthachk64-cyber.github.io, sedei-enterprise-integration, vr-fraudnet-reproduction; none concerns this paper). |

An unavailable paper artifact blocks an *exact reproduction*; it does not block working software. Every row below maps what the **abstract says** to a module, and marks each unresolved parameter as a companion default.

## Abstract claims -> modules

| Abstract element | Location | Meaning / units | Module or command | Status |
|---|---|---|---|---|
| Inputs: historical traffic, system events, intrusion logs, user-behaviour sequences | Abstract | raw telemetry | `tdam.data` (authentication events only) | **Partial.** Only the authentication modality is implemented; `dns`, `flow`, `process` are recorded as absent, never imputed. |
| Temporal preprocessing / feature extraction | Abstract | windows over time | `tdam.temporal.binning`, `tdam.temporal.features` | Companion default: 1-minute bins, W = 30 bins. |
| LSTM network | Abstract | sequence model | `tdam.models.lstm` | Companion default: 1 layer, 64 units, linear logit head. |
| Hybrid ensemble learning | Abstract | combine learners | `tdam.models.ensemble` | Companion default: convex blend of LSTM and gradient-boosted trees; weight from chronological out-of-fold data. |
| Adaptive learning | Abstract | online update | `tdam.training.adaptive` | Companion default: walk-forward refit/fine-tune on matured labels. |
| Real-time preprocessing | Abstract | streaming | `tdam.data.replay` | Implemented as arrival-order replay. |
| Accuracy, detection latency, threat anticipation beat traditional ML | Abstract | qualitative | `tdam.evaluation` | **No number published in the abstract.** Measured here only on synthetic data; not comparable. |
| Dataset, ensemble members, layer sizes, sequence length, horizon, update rule | Abstract | **unspecified** | `configs/default.yaml` | All marked `companion_default` in the config provenance map. |
| Equations, algorithms, tables, figures (Sections III-V) | Full text | - | - | **Unverified: full text not read.** If you hold the paper, add rows here and record any discrepancy. |

## Three kinds of result, kept apart

1. **Paper-reported:** `references/reported/paper_claims.yaml` (qualitative claims only).
2. **Independent benchmark:** LANL adapter implemented and tested on format fixtures; **no LANL run has been performed** (the data are not in the repository and were not downloaded).
3. **Synthetic demonstration:** `results/demo`, produced by `tdam demo`.
