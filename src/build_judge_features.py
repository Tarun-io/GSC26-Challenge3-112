"""
Phase 6a: build feature rows for the UNLABELED judge set.

Same windowing + aggregation logic as build_features.py, but with no
label construction -- the judge Parquet files have exitcode and
end_time stripped entirely (per README2.txt), so there is nothing to
build a label from and no exitcode column to even query.

Usage:
    python src/build_judge_features.py "data/judge_data" --cluster S --out outputs/judge_features_S.csv
"""

import argparse
import os

import duckdb
import pandas as pd

from build_features import load_config, cluster_filter_sql, build_value_column_sql


def build_judge_features(parquet_dir, cluster, cfg, window_pct=None):
    window_pct = window_pct if window_pct is not None else cfg["window_pct"]

    glob_pattern = os.path.join(parquet_dir, "**", "*.parquet")
    cluster_cond = cluster_filter_sql(cluster, cfg)
    value_sql = build_value_column_sql(cfg)

    query = f"""
    SELECT
      jid,
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
    GROUP BY jid
    """

    df = duckdb.sql(query).df()
    df["wait_time_sec"] = (df["start_time"] - df["submit_time"]).dt.total_seconds()

    print(f"[{cluster}] judge set: built {len(df)} job-level feature rows (no labels)")
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
    df = build_judge_features(args.parquet_dir, args.cluster, cfg, args.window_pct)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")