# Methodology Report — FRESCO Early Job Failure Prediction

## 1. Problem framing

Binary early-warning classification: predict, from only the early
telemetry of an HPC job, whether it will end in FAILED/TIMEOUT/NODE_FAIL
vs. COMPLETED. Scored via macro-averaged per-cluster AUPRC across three
clusters (Stampede/S, Conte/C, Anvil), modeled independently — no pooling.

## 2. Data and early-window rule

Raw hourly Parquet telemetry, one row per job per host per minute.
Jobs are grouped across files by `jid` (verified as a safe join key
via `src/check_jid_uniqueness.py` — see PROJECT_SPEC.md §2 for the
full jid-uniqueness check).

**Early-window rule:** per job, only telemetry with
`time <= start_time + (window_pct/100) * timelimit` is used, where
`window_pct = 10` (dynamic per job, not a flat wall-clock cutoff — see
PROJECT_SPEC.md §3 for the full design rationale, including why this
rule was chosen over alternatives). This choice is consistent with the
organizers' direct clarification that horizon `T` is a team-chosen
design decision (quoted in full in PROJECT_SPEC.md §8).

## 3. Leakage prevention

`exitcode` and `end_time` are never used as model features -- `exitcode`
is used exclusively to construct the training label. Full leakage rules
and reasoning: PROJECT_SPEC.md §2. Candidate features suspected of being
leakage-in-disguise (`n_rows_in_window`, `wait_time_sec`) were validated
via a correlation study on real, full-scale training data before being
trusted (PROJECT_SPEC.md, Phase 3 findings) -- both cleared with no
leakage signal found.

## 4. Feature engineering

One feature row per job: requested-resource metadata (timelimit,
nhosts, ncores, account, queue), queue wait time, and per-telemetry-
signal aggregates (mean/max/min/std/last-value/trend-slope) computed
only within the early window. Full feature list: PROJECT_SPEC.md §4.

## 5. Validation strategy

Time-respecting split, mirroring the organizers' own participant/
validation chunk boundaries where available. [UPDATE: as of this
report, the official Jan-Mar 2017 validation chunk was absent (cluster
S) or too small to trust (cluster C, 115 rows) in our downloaded data --
a fallback internal holdout (last 60 days of the training window,
still time-respecting) was used instead. Re-check whether an updated
validation release is available before final submission.]

## 6. Model

Gradient-boosted trees (XGBoost), one independent model per cluster,
categorical features (`account`, `queue`) handled via XGBoost's native
categorical support. Hyperparameters: [UPDATE if tuned beyond defaults
-- see PROJECT_SPEC.md §9 improvement backlog].

## 7. Results

| Cluster | Validation AUPRC | Validation source |
|---|---|---|
| S | 0.611 | fallback holdout |
| C | 0.672 | fallback holdout |
| Anvil | 0.844 | fallback holdout |

Macro AUPRC (local estimate): ~0.709.

[UPDATE this table if re-trained with different window_pct / hyperparameters.]

## 8. Known limitations / future work

See PROJECT_SPEC.md §9 for the full, current improvement backlog
(window sweep, hyperparameter tuning, class-imbalance handling,
smarter fallback for zero-telemetry jobs, v2 windowing).

## 9. AI/LLM usage disclosure

[REQUIRED BY COMPETITION RULES -- fill in accurately before submission.]

Large language model assistance (Claude) was used for: architecture
design discussion, code generation for the data pipeline (feature
engineering, train/val split, model training, submission assembly),
and debugging. All design decisions (window rule, leakage boundaries,
model choice) were discussed and reasoned through explicitly rather
than accepted blindly; generated code was tested against real sample
data at each stage before being scaled up (see verification steps
throughout PROJECT_SPEC.md). [Add/adjust based on your team's actual
process, per-team-member usage, and any other tools used.]