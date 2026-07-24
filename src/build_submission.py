"""
Phase 6b: score the judge set with the trained per-cluster models and
assemble the final submission.csv.

Handles the case where a job's early window produced zero telemetry
rows (so it never made it into judge_features_<cluster>.csv) -- every
required row_id in sample_submission.csv MUST get a prediction, so
missing jobs are filled with that cluster's own training failure rate
(a documented, defensible fallback, not a silent gap).

Usage:
    python src/build_submission.py
"""

import json

import pandas as pd
import xgboost as xgb

from train_model import CATEGORICAL_COLS

CLUSTERS = ["S", "C", "Anvil"]


def score_cluster(cluster, out_prefix="outputs"):
    model = xgb.XGBClassifier()
    model.load_model(f"{out_prefix}/model_{cluster}.json")

    with open(f"{out_prefix}/model_{cluster}_meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    cat_levels = meta["categorical_levels"]

    judge = pd.read_csv(
        f"{out_prefix}/judge_features_{cluster}.csv",
        dtype={c: str for c in CATEGORICAL_COLS},
    )

    for c in CATEGORICAL_COLS:
        if c in feature_cols:
            judge[c] = judge[c].astype("category").cat.set_categories(cat_levels[c])

    missing_feature_cols = [c for c in feature_cols if c not in judge.columns]
    if missing_feature_cols:
        print(f"WARNING [{cluster}]: judge features missing columns the model "
              f"expects: {missing_feature_cols} -- filling with NaN")
        for c in missing_feature_cols:
            judge[c] = float("nan")

    probs = model.predict_proba(judge[feature_cols])[:, 1]

    train = pd.read_csv(f"{out_prefix}/{cluster}_train.csv")
    fallback_prob = float(train["label"].mean())

    scored = pd.DataFrame({
        "row_id": cluster + "_" + judge["jid"].astype(str),
        "failure_probability": probs,
    })
    print(f"[{cluster}] scored {len(scored)} jobs, fallback rate for missing jobs: {fallback_prob:.4f}")
    return scored, fallback_prob


def build_submission(sample_submission_path="data/sample_submission.csv",
                      out_prefix="outputs", out_path="outputs/submission.csv"):
    sample = pd.read_csv(sample_submission_path)
    required_ids = set(sample["row_id"])
    print(f"required row_ids: {len(required_ids)}")

    all_scored = []
    fallback_used_total = 0
    for cluster in CLUSTERS:
        scored, fallback_prob = score_cluster(cluster, out_prefix)
        all_scored.append(scored)

        cluster_required = {r for r in required_ids if r.startswith(f"{cluster}_")}
        scored_ids = set(scored["row_id"])
        missing = cluster_required - scored_ids
        print(f"[{cluster}] {len(missing)} required row_ids had no early-window "
              f"telemetry -- filling with fallback prob {fallback_prob:.4f}")

        if missing:
            fallback_rows = pd.DataFrame({
                "row_id": list(missing),
                "failure_probability": fallback_prob,
            })
            all_scored.append(fallback_rows)
            fallback_used_total += len(missing)

    submission = pd.concat(all_scored, ignore_index=True)

    # keep only required ids, dedupe (model-scored row wins over fallback
    # if a row_id somehow ended up in both), enforce exact match
    submission = submission[submission["row_id"].isin(required_ids)]
    submission = submission.drop_duplicates(subset="row_id", keep="first")

    missing_after = required_ids - set(submission["row_id"])
    extra_after = set(submission["row_id"]) - required_ids

    print(f"total fallback predictions used: {fallback_used_total}")
    print(f"final submission rows: {len(submission)} (required: {len(required_ids)})")
    print(f"missing after assembly: {len(missing_after)}, extra after assembly: {len(extra_after)}")

    assert len(missing_after) == 0, "submission is missing required row_ids -- do not submit"
    assert len(extra_after) == 0, "submission has extra row_ids not in sample_submission -- do not submit"
    assert len(submission) == len(required_ids), "row count mismatch -- do not submit"

    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    build_submission()