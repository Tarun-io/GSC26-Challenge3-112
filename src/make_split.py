"""
Phase 4: time-respecting train/validation split.

Assigns each job to 'train' or 'val' based on its start_time falling
inside the organizer's own participant/validation chunk date ranges
(config.yml -> split_ranges). If the official validation slice is
missing or too small/single-class to trust for AUPRC, falls back to a
time-respecting internal holdout carved from the tail of the training
window instead -- never a random shuffle.

Usage:
    python src/make_split.py outputs/features_S_full.csv --cluster S
"""

import argparse

import pandas as pd
import yaml


def load_config(path="config.yml"):
    with open(path) as f:
        return yaml.safe_load(f)


MIN_TRUSTED_VAL_ROWS = 500  # below this, AUPRC is too noisy to trust


def make_split(csv_path, cluster, cfg, fallback_holdout_days=60):
    df = pd.read_csv(csv_path, parse_dates=["start_time"])
    n_missing_start = df["start_time"].isna().sum()
    if n_missing_start:
        print(f"[{cluster}] dropping {n_missing_start} rows with missing start_time")
        df = df.dropna(subset=["start_time"])

    ranges = cfg["split_ranges"][cluster]

    train_mask = (df["start_time"] >= ranges["train_start"]) & (df["start_time"] <= ranges["train_end"])
    val_mask = (df["start_time"] >= ranges["val_start"]) & (df["start_time"] <= ranges["val_end"])

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    unassigned = df[~(train_mask | val_mask)]

    print(f"[{cluster}] total rows: {len(df)}")
    print(f"[{cluster}] official train: {len(train_df)} rows "
          f"({ranges['train_start']} to {ranges['train_end']})")
    print(f"[{cluster}] official val:   {len(val_df)} rows "
          f"({ranges['val_start']} to {ranges['val_end']})")
    print(f"[{cluster}] unassigned (outside both ranges): {len(unassigned)}")

    official_val_trustworthy = (
        len(val_df) >= MIN_TRUSTED_VAL_ROWS and val_df["label"].nunique() == 2
    )

    if official_val_trustworthy:
        print(f"[{cluster}] official validation set is usable as-is "
              f"({len(val_df)} rows, label=1: {int(val_df['label'].sum())} "
              f"[{val_df['label'].mean():.1%}])")
        final_train, final_val = train_df, val_df
        source = "official"
    else:
        # Fallback: carve the LAST fallback_holdout_days of the official
        # training window as a local, time-respecting proxy validation
        # set. This is not the organizer's real validation chunk -- it's
        # a stand-in so we can still measure AUPRC locally until/unless
        # the real validation data becomes available.
        print(f"[{cluster}] official validation set is missing or too small "
              f"(<{MIN_TRUSTED_VAL_ROWS} rows or single-class) -- "
              f"falling back to a {fallback_holdout_days}-day internal holdout "
              f"from the tail of the training window.")
        cutoff = pd.Timestamp(ranges["train_end"]) - pd.Timedelta(days=fallback_holdout_days)
        local_val_mask = train_df["start_time"] > cutoff
        final_val = train_df[local_val_mask].copy()
        final_train = train_df[~local_val_mask].copy()
        source = "fallback"
        print(f"[{cluster}] fallback train: {len(final_train)} rows "
              f"(up to {cutoff.date()}), label=1: {int(final_train['label'].sum())} "
              f"({final_train['label'].mean():.1%})")
        if len(final_val) > 0:
            print(f"[{cluster}] fallback val:   {len(final_val)} rows "
                  f"(after {cutoff.date()}), label=1: {int(final_val['label'].sum())} "
                  f"({final_val['label'].mean():.1%})")
        else:
            print(f"[{cluster}] fallback val:   0 rows (after {cutoff.date()})")

    if len(final_val) == 0:
        print(f"WARNING [{cluster}]: still empty validation set even after fallback -- "
              f"cannot compute AUPRC. Needs manual attention.")
    elif final_val["label"].nunique() < 2:
        print(f"WARNING [{cluster}]: validation set has only one class present -- "
              f"AUPRC is undefined without both a positive and negative label.")

    if len(unassigned) > 0:
        print(f"NOTE [{cluster}]: {len(unassigned)} rows fell outside the expected "
              f"date ranges -- worth spot-checking a few before assuming this is fine.")

    print(f"[{cluster}] validation source used: {source}")
    return final_train, final_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--cluster", required=True, choices=["S", "C", "Anvil"])
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--out-prefix", default=None,
                         help="output prefix, default: outputs/<cluster>")
    parser.add_argument("--fallback-holdout-days", type=int, default=60,
                         help="days held out from tail of train window when "
                              "official val is missing/too small")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_df, val_df = make_split(args.csv_path, args.cluster, cfg, args.fallback_holdout_days)

    prefix = args.out_prefix or f"outputs/{args.cluster}"
    train_df.to_csv(f"{prefix}_train.csv", index=False)
    val_df.to_csv(f"{prefix}_val.csv", index=False)
    print(f"wrote {prefix}_train.csv and {prefix}_val.csv")