"""
Diagnostic: checks whether any `jid` value spans a suspiciously long time
range within one cluster. If job IDs get reused over years, GROUP BY jid
would silently merge two unrelated jobs into one garbage feature row --
this script catches that before it corrupts anything downstream.

Usage:
    python src/check_jid_uniqueness.py "data/training_data" --cluster S
"""

import argparse
import os

import duckdb
import yaml


def load_config(path="config.yml"):
    with open(path) as f:
        return yaml.safe_load(f)


def cluster_filter_sql(cluster, cfg):
    suffix = cfg["clusters"].get(cluster)
    if suffix:
        return f"filename LIKE '%{suffix}'"
    others = [s for s in cfg["clusters"].values() if s]
    return " AND ".join(f"filename NOT LIKE '%{s}'" for s in others)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_dir")
    parser.add_argument("--cluster", required=True, choices=["S", "C", "Anvil"])
    parser.add_argument("--config", default="config.yml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    glob_pattern = os.path.join(args.parquet_dir, "**", "*.parquet")
    cluster_cond = cluster_filter_sql(args.cluster, cfg)

    query = f"""
    WITH per_job AS (
      SELECT
        jid,
        MIN(start_time) AS min_start,
        MAX(start_time) AS max_start,
        COUNT(DISTINCT submit_time) AS n_distinct_submits
      FROM read_parquet('{glob_pattern}', filename=true)
      WHERE {cluster_cond}
      GROUP BY jid
    )
    SELECT
      COUNT(*) AS total_jobs,
      SUM(CASE WHEN date_diff('day', min_start, max_start) > 2 THEN 1 ELSE 0 END) AS suspicious_jobs,
      MAX(date_diff('day', min_start, max_start)) AS max_span_days
    FROM per_job
    """

    result = duckdb.sql(query).df()
    print(result.to_string(index=False))

    n_suspicious = int(result["suspicious_jobs"].iloc[0])
    if n_suspicious > 0:
        print(f"\nWARNING: {n_suspicious} jid values span more than 2 days -- "
              f"likely job-ID reuse merging unrelated jobs. Do not train on "
              f"the current feature tables until this is resolved.")
    else:
        print("\nNo suspicious jid spans found -- jid appears safe to use as a join key.")