"""
FRESCO early-job-failure feature builder (DuckDB version).

Recursively reads every Parquet file under a directory (including nested
month subfolders) as ONE queryable table, so a job whose telemetry spans
multiple hourly files -- or files across nested folders -- is grouped
correctly by `jid` in a single SQL query. No manual cross-file stitching.

All rules (leakage columns, window %, feature list) come from config.yml,
never hardcoded here. See PROJECT_SPEC.md for the reasoning behind each
rule.

Usage:
    python src/build_features.py "data/training_data" --cluster S --out outputs/features_S.csv
"""

import argparse
import os

import duckdb
import pandas as pd
import yaml


def load_config(path="config.yml"):
    with open(path) as f:
        return yaml.safe_load(f)


def cluster_filter_sql(cluster, cfg):
    """Build a WHERE-clause fragment that filters rows to one cluster,
    based on the source filename (works no matter how deeply nested the
    file is in month subfolders)."""
    suffix = cfg["clusters"].get(cluster)
    if suffix:
        return f"filename LIKE '%{suffix}'"
    others = [s for s in cfg["clusters"].values() if s]
    return " AND ".join(f"filename NOT LIKE '%{s}'" for s in others)


def build_value_column_sql(cfg):
    expr_map = {
        "mean": "AVG({c})",
        "max": "MAX({c})",
        "min": "MIN({c})",
        "std": "STDDEV({c})",
        "last": "ARG_MAX({c}, epoch(time))",
        "slope": "REGR_SLOPE({c}, epoch(time))",
    }
    parts = []
    for c in cfg["value_columns"]:
        for agg in cfg["aggregations"]:
            alias = f"{c}_{agg}"
            parts.append(f"{expr_map[agg].format(c=c)} AS {alias}")
    return ",\n  ".join(parts)


def build_features(parquet_dir, cluster, cfg, window_pct=None):
    window_pct = window_pct if window_pct is not None else cfg["window_pct"]

    # recursive glob -- reaches files nested inside month subfolders
    glob_pattern = os.path.join(parquet_dir, "**", "*.parquet")
    cluster_cond = cluster_filter_sql(cluster, cfg)
    dropped_sql = ",".join(f"'{s}'" for s in cfg["dropped_states"])
    value_sql = build_value_column_sql(cfg)

    query = f"""
    SELECT
      jid,
      ANY_VALUE(exitcode)     AS exitcode,
      ANY_VALUE(start_time)   AS start_time,
      ANY_VALUE(submit_time)  AS submit_time,
      ANY_VALUE(timelimit)    AS timelimit_min,
      ANY_VALUE(nhosts)       AS nhosts,
      ANY_VALUE(ncores)       AS ncores,
      ANY_VALUE(account)      AS account,
      ANY_VALUE(queue)        AS queue,
      COUNT(*)                AS n_rows_in_window,
      COUNT(DISTINCT host)    AS n_hosts_in_window,
      {value_sql}
    FROM read_parquet('{glob_pattern}', filename=true)
    WHERE {cluster_cond}
      AND epoch(time) <= epoch(start_time) + timelimit * 60 * ({window_pct} / 100.0)
      AND exitcode NOT IN ({dropped_sql})
    GROUP BY jid
    """

    df = duckdb.sql(query).df()

    positive = set(cfg["positive_states"])
    negative = set(cfg["negative_states"])
    df["label"] = df["exitcode"].map(lambda e: 1 if e in positive else (0 if e in negative else None))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["wait_time_sec"] = (df["start_time"] - df["submit_time"]).dt.total_seconds()

    for banned in cfg["banned_feature_columns"]:
        assert banned not in df.columns or banned == cfg["label_column"], (
            f"leakage column '{banned}' leaked into feature table"
        )

    print(f"[{cluster}] built {len(df)} job-level feature rows "
          f"(label=1: {int(df['label'].sum())}, label=0: {int((df['label']==0).sum())})")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_dir")
    parser.add_argument("--cluster", required=True, choices=["S", "C", "Anvil"])
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--window-pct", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    feat_df = build_features(args.parquet_dir, args.cluster, cfg, args.window_pct)
    feat_df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")