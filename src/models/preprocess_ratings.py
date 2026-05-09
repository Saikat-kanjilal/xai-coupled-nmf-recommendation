from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    guess_column,
    make_parser,
    normalize_columns,
    require_file,
    setup_logging,
    validate_dataset_name,
    write_csv,
)


def preprocess_ratings(
    dataset: str,
    input_ratings: str,
    output_csv: str,
    min_user_ratings: int = 20,
    min_item_ratings: int = 20,
    log_file: str | None = None,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    require_file(input_ratings)

    df = pd.read_csv(input_ratings)
    df = normalize_columns(df)

    user_col = guess_column(df, ["user_id", "userid", "user"])
    item_col = guess_column(df, ["item_id", "movie_id", "movieid", "item"])
    rating_col = guess_column(df, ["rating", "ratings", "score"])
    timestamp_col = guess_column(df, ["timestamp", "time"], required=False)

    keep = [user_col, item_col, rating_col] + ([timestamp_col] if timestamp_col else [])
    out = df[keep].copy()
    out.columns = ["user_id", "item_id", "rating"] + (["timestamp"] if timestamp_col else [])

    out = out.dropna(subset=["user_id", "item_id", "rating"]).drop_duplicates(subset=["user_id", "item_id"]) 
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out = out.dropna(subset=["rating"])
    out = out[out["rating"] >= 0]

    user_counts = out.groupby("user_id")["item_id"].count()
    keep_users = set(user_counts[user_counts >= min_user_ratings].index)
    out = out[out["user_id"].isin(keep_users)]

    item_counts = out.groupby("item_id")["user_id"].count()
    keep_items = set(item_counts[item_counts >= min_item_ratings].index)
    out = out[out["item_id"].isin(keep_items)]

    if "timestamp" not in out.columns:
        out["timestamp"] = ""

    out = out[["user_id", "item_id", "rating", "timestamp"]].sort_values(["user_id", "item_id"]).reset_index(drop=True)
    write_csv(out, output_csv)
    logger.info("Processed ratings written to %s with %d rows", output_csv, len(out))


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Preprocess raw ratings into manuscript workflow format.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_ratings", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--min_user_ratings", type=int, default=20)
    p.add_argument("--min_item_ratings", type=int, default=20)
    p.add_argument("--log_file", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    preprocess_ratings(**vars(args))


if __name__ == "__main__":
    main()
