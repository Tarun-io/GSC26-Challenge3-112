"""
Phase 5: train one gradient-boosted model per cluster.

Reads the train/val split built in Phase 4 (outputs/<cluster>_train.csv,
outputs/<cluster>_val.csv), trains an XGBoost classifier, and reports
AUPRC on the validation set -- the SAME metric the competition scores
on (average_precision_score per cluster).

No pooling across clusters -- this script is run once per cluster,
producing one independent model per cluster, per the competition rule.

Usage:
    python src/train_model.py --cluster S
"""

import argparse
import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

DROP_COLS = ["jid", "exitcode", "start_time", "submit_time", "label"]
CATEGORICAL_COLS = ["account", "queue"]


def load_split(cluster, out_prefix):
    dtype_overrides = {c: str for c in CATEGORICAL_COLS}
    train = pd.read_csv(f"{out_prefix}/{cluster}_train.csv", dtype=dtype_overrides)
    val = pd.read_csv(f"{out_prefix}/{cluster}_val.csv", dtype=dtype_overrides)

    feature_cols = [c for c in train.columns if c not in DROP_COLS]

    # fit categorical levels from TRAIN only, apply the same levels to val
    # (and later, the judge set) -- unseen categories become NaN, which
    # XGBoost's native categorical support handles natively.
    cat_levels = {}
    for c in CATEGORICAL_COLS:
        if c in feature_cols:
            train[c] = train[c].astype("category")
            cat_levels[c] = list(train[c].cat.categories)
            val[c] = val[c].astype("category").cat.set_categories(cat_levels[c])

    return train, val, feature_cols, cat_levels


def train_and_eval(cluster, out_prefix="outputs"):
    train, val, feature_cols, cat_levels = load_split(cluster, out_prefix)

    X_train, y_train = train[feature_cols], train["label"]
    X_val, y_val = val[feature_cols], val["label"]

    print(f"[{cluster}] train: {len(X_train)} rows, {len(feature_cols)} features")
    print(f"[{cluster}] val:   {len(X_val)} rows")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    auprc = average_precision_score(y_val, val_probs)
    print(f"[{cluster}] VALIDATION AUPRC: {auprc:.4f}")

    model_path = f"{out_prefix}/model_{cluster}.json"
    model.save_model(model_path)

    meta_path = f"{out_prefix}/model_{cluster}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "cluster": cluster,
            "feature_cols": feature_cols,
            "categorical_levels": cat_levels,
            "val_auprc": auprc,
        }, f, indent=2)

    print(f"[{cluster}] wrote {model_path} and {meta_path}")
    return auprc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", required=True, choices=["S", "C", "Anvil"])
    parser.add_argument("--out-prefix", default="outputs")
    args = parser.parse_args()

    train_and_eval(args.cluster, args.out_prefix)