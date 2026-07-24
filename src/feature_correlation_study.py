"""
Feature correlation study -- run on the REAL feature tables (millions of
rows), not the 28-row toy sample. This is the gate that decides which
candidate features (esp. n_rows_in_window, wait_time_sec) are trustworthy
signal vs. accidental leakage-in-disguise, using actual statistical
evidence instead of a guess from a tiny sample.

Usage:
    python src/feature_correlation_study.py outputs/features_S_full.csv --cluster S
"""

import argparse

import numpy as np
import pandas as pd


def run_study(csv_path, cluster, top_n=15):
    df = pd.read_csv(csv_path)
    print(f"[{cluster}] loaded {len(df)} rows, label balance: "
          f"1={int(df['label'].sum())} ({df['label'].mean():.1%}), "
          f"0={int((df['label']==0).sum())}")

    numeric_df = df.select_dtypes(include=[np.number])
    candidate_cols = [c for c in numeric_df.columns if c != "label"]

    results = []
    for c in candidate_cols:
        sub = df[["label", c]].dropna()
        if sub[c].nunique() < 2 or len(sub) < 30:
            continue
        corr = sub["label"].corr(sub[c])
        results.append((c, corr, len(sub), len(sub) / len(df)))

    res_df = pd.DataFrame(
        results, columns=["feature", "corr_with_label", "n_valid", "coverage"]
    ).sort_values("corr_with_label", key=abs, ascending=False)

    print(f"\nTop {top_n} features by |correlation with label|:")
    print(res_df.head(top_n).to_string(index=False))

    print("\nCandidate features flagged for leakage-proxy suspicion:")
    for watch in ["n_rows_in_window", "wait_time_sec", "n_hosts_in_window"]:
        row = res_df[res_df["feature"] == watch]
        if not row.empty:
            print(row.to_string(index=False))

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    run_study(args.csv_path, args.cluster, args.top_n)