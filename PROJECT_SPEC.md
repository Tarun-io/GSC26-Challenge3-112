# FRESCO Early Job Failure Prediction — Project Spec & Checklist

Context for any AI/teammate picking this up cold: this is the IEEE Computer
Society 2026 Global Student Challenge (Kaggle). Task: predict, from only the
*early* telemetry of an HPC job, the probability it will end in
FAILED/TIMEOUT/NODE_FAIL rather than COMPLETED. Three clusters (Stampede=S,
Conte=C, Anvil=no-suffix) must be modeled with **separate models/heads** —
pooling is explicitly disallowed. Scored with macro-averaged per-cluster
AUPRC (not accuracy). One prediction per unique `(cluster, jid)` job, output
as `submission.csv` inside a zip, row_id format `<cluster>_<jid>`.

Data scale: participant chunk ~46,795 hourly Parquet files, ~7GB. Judge set
(what we're scored against) = 654,155 required rows: C=348,275,
Anvil=209,943, S=95,937 (counted directly from sample_submission.csv).

---

## 1. Confirmed raw schema (verified from actual sample file, cluster S)

One row = one telemetry reading for one job, on one host, at one minute.
A single job can have hundreds of rows within one hourly file if it spans
many hosts (`nhosts` can be in the hundreds for large parallel jobs).

| column | type | role |
|---|---|---|
| `time` | datetime | reading timestamp — SAFE, defines window |
| `submit_time` | datetime | job queued — SAFE |
| `start_time` | datetime | job began running — SAFE |
| `end_time` | datetime | job finished — **LEAK, drop** (also stripped from real judge files per README2.txt) |
| `timelimit` | int (minutes) | requested wall time — SAFE, static per job |
| `nhosts` | int | requested host count — SAFE |
| `ncores` | int | requested core count — SAFE |
| `account` | str | submitting account — SAFE |
| `queue` | str | queue name — SAFE |
| `host` / `host_list` | str | node identity — SAFE |
| `jid` | str | job id — join key, not a feature |
| `unit` | str | metadata — SAFE, likely low value |
| `jobname` | str | free text — SAFE, likely low value |
| `exitcode` | str | final state (COMPLETED/FAILED/TIMEOUT/CANCELLED/NODE_FAIL) — **LABEL ONLY, never a feature** |
| `username` | str | SAFE |
| `value_cpuuser` | float | SAFE inside window only |
| `value_gpu` | float | SAFE inside window only |
| `value_memused` | float | SAFE inside window only |
| `value_memused_minus_diskcache` | float | SAFE inside window only |
| `value_nfs` | float | SAFE inside window only |
| `value_block` | float | SAFE inside window only |

**Label construction:** `y = 1` if `exitcode` in {FAILED, TIMEOUT, NODE_FAIL},
`y = 0` if `exitcode == COMPLETED`. `CANCELLED` jobs are excluded from the
positive class per the rules; current working decision is to **drop
CANCELLED from training** (candidate to revisit — could also try training
with them as `y=0` and compare validation AUPRC).

---

## 2. Leakage rules (locked)

- Never use `exitcode` or `end_time` as a model input feature.
- Never use anything derived from the job's true total duration
  (`end_time - start_time`) as a feature.
- Only use telemetry rows whose `time` falls inside the job's early window
  (see §3). Do not aggregate over the job's full lifetime.
- Be suspicious of any engineered feature that indirectly encodes "how much
  data exists for this job" as a proxy for "did it die early" — validate
  such features via correlation study on labeled (participant) data before
  trusting them; don't assume guilty or innocent by default.

## 3. Early-window rule (locked, v1)

Per job, keep only telemetry rows where:

```
time <= start_time + X% * timelimit(minutes)
```

- `X` = tunable, start at **10%**, sweep 5–20% during validation.
- Dynamic per job (scales with each job's own requested `timelimit`), not a
  flat wall-clock cutoff — avoids penalizing/favoring jobs of different
  requested sizes.
- If a job dies before the window closes, just use however many rows it
  actually produced — this is not a leak, since the window boundary was
  set from `timelimit` (known at submission), not from the job's actual
  outcome/duration.

**v2 upgrade (planned, not yet built):** replace flat `X%` with a window
sized from a *learned typical runtime* computed from historical jobs
sharing similar `account`/`queue`/`ncores` profile (Mode-A-only feature,
non-leaky since it's built from *other* jobs' history, not this job's own
future). Requires v1 pipeline working first to have a baseline to compare
against.

## 4. Aggregation / "Profiler" step

Collapse each job's in-window rows (many rows × many hosts) into **one
fixed-shape feature row**. Planned features per job:

- `wait_time` = `start_time - submit_time` (seconds)
- for each of `value_cpuuser`, `value_gpu`, `value_memused`,
  `value_memused_minus_diskcache`, `value_nfs`, `value_block`:
  mean, max, std, min, last-value, simple trend (slope of value vs time)
- `n_rows_in_window` (candidate feature — validate via correlation study,
  don't assume safe or unsafe)
- `n_hosts_reporting` (host spread within window)
- `ncores`, `nhosts`, `timelimit` (raw, static)
- categorical: `account`, `queue` (target/frequency encode later)

## 5. Modeling

- One gradient-boosted tree model (XGBoost or LightGBM) **per cluster**
  (S, C, Anvil) — no pooled model.
- Router at inference: dispatch each `row_id` to its cluster's model based
  on `_S` / `_C` / no-suffix.
- Deep learning (LSTM/Transformer on raw sequences) is a possible stretch
  goal *after* the boosted-tree baseline works end-to-end — not the
  primary architecture, given team's current coding level and competition
  timeline.

## 6. Validation

- Time-respecting split, mirroring the organizer's own chunking (don't
  random-shuffle across time). Train on early portion of participant data,
  validate on a later held-out slice, consistent with how
  participant → validation → judge chunks are ordered in
  `data-chunk-info.txt`.
- Metric to track locally: per-cluster AUPRC (`average_precision_score`),
  then macro-average across the three — matches the actual competition
  metric exactly, don't just track accuracy.
- Each cluster's validation slice must contain at least one positive and
  one negative label (required by the metric definition).

## 7. Submission mechanics

- Output: `submission.csv` with columns `row_id, failure_probability`,
  one row per required `row_id` in `sample_submission.csv` (654,155 rows
  total: C=348,275 / Anvil=209,943 / S=95,937).
- Zip must contain **only** `submission.csv` — no metadata files.
- Separately: private GitHub repo (README, deps file, training + inference
  code, methodology doc) — required all challenge, not a Kaggle upload.

---

## 8. Official organizer clarification (received directly, not inferred)

The organizers confirmed the following in direct correspondence — this
should be quoted/paraphrased in the methodology report as justification
for our windowing and feature design:

- There is **no fixed organizer-prescribed cutoff** after `start_time`.
  Each team chooses its own early horizon `T` and must document the
  choice and how it's enforced. Our choice: `T` = a dynamic per-job
  window, `10%` of that job's own `timelimit` (see §3) — not a flat
  wall-clock number, and explicitly allowed under this rule.
- At the decision point `start_time + T`, allowed inputs are: (a)
  start/submit-time metadata (requested resources, queue/cluster
  context, timelimit), and (b) telemetry with timestamp `<= start_time + T`.
- **Explicitly allowed**: last observed timestamp, record counts, and
  summaries within the early window. This directly confirms
  `n_rows_in_window`, `n_hosts_in_window`, and the `*_last` aggregation
  features are legitimate, not leakage-in-disguise — consistent with
  what our Phase 3 correlation study already found (no leakage red flag
  on any of these, see §Phase 3 findings).
- **Explicitly disallowed**: features based on a job's full lifetime,
  "last telemetry ever," telemetry stopping, or near-termination
  measurements — since these implicitly encode that the job has ended.
  Our pipeline is compliant by construction: the window filter
  (`time <= start_time + window`) means rows after a job's actual end
  never exist in the source data to begin with, so there is no
  mechanism by which a "job already ended" signal could enter the
  feature table.

**Conclusion: no pipeline changes required.** This confirms the
windowing design in §3 and the feature set validated in Phase 3 were
already compliant before this clarification was received.

## 9. Improvement backlog (deferred — revisit after initial judge review)

Current state is a genuinely working, rule-compliant baseline (Phases 1-7
complete, submission.csv verified against sample_submission.csv exactly).
These are known, real opportunities to improve score, deliberately
deferred rather than done now:

1. **Window `X%` was never swept.** Locked at 10% by default, never
   tested against 5/15/20% as planned back in Phase 3. Likely the
   single highest-leverage remaining improvement.
2. **XGBoost hyperparameters are untuned defaults**
   (`n_estimators=300, max_depth=6, lr=0.05`) — never searched.
3. **Class imbalance not explicitly handled** — especially cluster C
   (~8-13% positive rate). `scale_pos_weight` untried.
4. **2,765 cluster-S jobs (~2.9%) fall back to a flat cluster base rate**
   because their early window produced zero telemetry rows. A smarter
   fallback (e.g. account/queue-specific typical rate instead of whole-
   cluster rate) may do better than the current flat guess.
5. **v2 windowing idea (learned typical runtime per account/queue,
   discussed and designed together) was never implemented** — still a
   documented idea only, not code.
6. **Local AUPRC is measured on a fallback validation slice** (tail of
   training data), not the organizer's real Jan-Mar 2017 validation
   chunk (which is absent/too-small in the current download) — treat
   current AUPRC numbers (S: 0.611, C: 0.672, Anvil: 0.844) as estimates,
   not guarantees of real leaderboard score.

## Checklist — status

- [x] Understand rules, metric, target definition
- [x] Inspect real Parquet schema (cluster S sample file)
- [x] Identify and lock leakage rules (`exitcode`, `end_time`)
- [x] Design dynamic, essence-based early-window rule (v1: flat % of
      `timelimit`; v2 planned: learned-runtime-based)
- [x] Decide model family (gradient-boosted trees, per-cluster)
- [x] Build ingestor: loop over all participant Parquet files per cluster
      (DuckDB, recursive glob, cross-file jid-safe — verified via
      check_jid_uniqueness.py)
- [x] Build profiler/aggregation function (one row per job)
- [x] Correlation study: validate candidate features (esp. `n_rows_in_window`)
      against labels on participant data — cleared, no leakage signal
- [x] Decide final CANCELLED-handling policy (drop from training) — locked
- [x] Train per-cluster XGBoost baseline (S: 0.611, C: 0.672,
      Anvil: 0.844 validation AUPRC, macro ~0.709)
- [x] Time-respecting validation split + per-cluster AUPRC check
      (fallback holdout used — official Jan-Mar 2017 chunk absent/too
      small in current download, see §Phase 4 findings)
- [ ] Sweep window `X%` (5–20%) using validation AUPRC — deferred, see §9
- [x] Build inference/router script producing `submission.csv`
- [x] Verify submission covers all 654,155 required row_ids exactly —
      confirmed, zero missing/extra, matches per-cluster counts exactly
- [ ] Set up private GitHub repo, invite organizer accounts, add
      README/deps/methodology doc — in progress