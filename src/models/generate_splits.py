from __future__ import annotations

import argparse

import pandas as pd

from common import make_parser, parse_int_list, read_csv, setup_logging, validate_dataset_name, validate_split_type, write_csv


def _split_user_frame(df: pd.DataFrame, split_type: str, seed: int) -> pd.DataFrame:
    if split_type == "chronological_per_user" and "timestamp" in df.columns:
        ordered = df.sort_values(["user_id", "timestamp", "item_id"]).copy()
    else:
        ordered = df.sample(frac=1.0, random_state=seed).copy()

    parts = []
    for user_id, group in ordered.groupby("user_id", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        n_train = max(int(round(0.8 * n)), 1)
        n_val = max(int(round(0.1 * n)), 1) if n >= 3 else 0
        n_test = max(n - n_train - n_val, 1 if n >= 2 else 0)
        if n_train + n_val + n_test > n:
            n_train = max(n_train - 1, 1)
        split = ["train"] * n
        for i in range(n_train, min(n_train + n_val, n)):
            split[i] = "val"
        for i in range(min(n_train + n_val, n), n):
            split[i] = "test"
        group["split"] = split
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def generate_splits(
    dataset: str,
    input_ratings: str,
    split_type: str,
    seed_list: str,
    output_csv: str,
    log_file: str | None = None,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    validate_split_type(split_type)
    seeds = parse_int_list(seed_list)
    ratings = read_csv(input_ratings)
    if "timestamp" not in ratings.columns:
        ratings["timestamp"] = ""

    parts = []
    for seed in seeds:
        split_df = _split_user_frame(ratings[["user_id", "item_id", "rating", "timestamp"]].copy(), split_type, seed)
        split_df.insert(0, "split_type", split_type)
        split_df.insert(0, "seed", seed)
        split_df.insert(0, "dataset_name", dataset)
        parts.append(split_df)
    out = pd.concat(parts, ignore_index=True)
    write_csv(out, output_csv, schema_name="split_registry")
    logger.info("Split registry written to %s with %d rows", output_csv, len(out))


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Generate per-user train/val/test split registry.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_ratings", required=True)
    p.add_argument("--split_type", required=True)
    p.add_argument("--seed_list", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--log_file", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    generate_splits(**vars(args))


if __name__ == "__main__":
    main()
