# Architecture and forecast contract

## Contract
Decision time *t*, entity *e* (a pseudonymous host), lookback *W* bins, horizon *H*.

* **Inputs:** records with event time in (t-W, t] **and** arrival time <= t. A record stamped 10:16 but ingested 10:18 is invisible to the 10:17 decision.
* **Target:** y = 1 iff an annotated incident onset of *e* lies in (t, t+H]. An onset equal to *t* is not in the horizon.
* **Exclusions:** ACTIVE (incident in progress), RECOVERY (ended within R minutes), CENSORED (future window not fully observed). Unmonitored times are not decision points.
* **Separate times:** onset, end, label_available per incident; issue time per alert.
* **Companion defaults:** 1-minute bins, W = 30, H = 5 min, R = 10 min, alert cooldown 10 min.

Worked examples (incident onset 10:20 UTC), each a regression test in `tests/test_visibility.py`: alert 10:17 -> 3 min lead; alert 10:21 -> 1 min detection delay and zero early credit; record 10:16 ingested 10:18 -> invisible at 10:17.

## Splits
Chronological `train | val | calib | test` = 50/10/10/30 % of the timeline. A window belongs to a partition only if [t-W+1, t+H] lies wholly inside it (**purge**). History is **not** carried across partition boundaries in the fixed holdout; the online replay is causal and does carry state. An incident belongs to the single partition whose decision span contains its onset; others are `purged` (counted in `partitions.json`).
Fit on train only: scaler, LSTM, tree, logistic, prevalence. Val: LSTM early stopping. Out-of-fold (expanding windows inside train): blend weight. Calib: thresholds. Test: nothing is fitted or chosen on it.

## Model (reference implementation)
```
x [B, 30, 6]  channels: events, fail, success, uniq_dst, gap_min, observed
   -> log1p + train-fitted standardisation (padding stays exactly 0)
   -> shift padding behind data, pack_padded_sequence(lengths)
   -> LSTM(6 -> 64, 1 layer, unidirectional); final valid hidden state [B, 64]
   -> Linear(64 -> 1) logit
loss: BCEWithLogits, pos_weight = n_neg/n_pos from training windows; Adam 1e-3; early stop on val AP
tree: HistGradientBoosting on 16 window summaries (recent-vs-prior change features), class_weight balanced
ensemble: w * p_lstm + (1-w) * p_tree, w in [0,1] from out-of-fold average precision
```
Class order: 0 = no annotated onset in (t, t+H]; 1 = onset in (t, t+H]. Scores are ranking scores, not calibrated probabilities. `observed = 1` with zero counts is a **true zero**; `observed = 0` is **unknown coverage**.

## Alerting and matching
Per-model threshold chosen on calib: lowest threshold with at most B unmatched warnings per monitored host-hour (B = 0.03), plus an F1-optimal threshold. Crossings inside the cooldown are grouped; an issued alert keeps its decision timestamp. Matching (`evaluation/matching.py`): EARLY (before next onset, within H, earliest alert only) > LATE (inside an active incident, no earlier credit) > REDUNDANT > UNMATCHED. Overlapping incidents prefer an unclaimed upcoming onset, then an unclaimed active one. "Unmatched" means unmatched to *known labels*.

## Adaptation (walk-forward)
Score first, update later. At each scheduled update (default every 12 h) only windows with `known_at <= now` are used, where `known_at` = horizon elapsed AND every incident that began by t+H already delivered by the annotation feed. Tree refit on train + matured; LSTM fine-tuned on matured + a replay sample. Scaler, blend weight and thresholds stay frozen. Every update is logged (`adaptive_updates.json`) with its version and the largest `known_at` used.

## Replay
`tdam.data.replay` ingests by arrival time; late records inside the window are folded in for later ticks only; stale records are dropped and counted; duplicate ids are ignored while their bin is in the window; entity state is pruned after the window (bounded memory); idle entities are scored with an empty window. Replay and batch scores are identical (tested, max abs diff < 1e-5).

## Differences from the verified paper
None can be verified (full text unread). Every architectural choice above is a companion default, labelled as such in `configs/default.yaml`.
