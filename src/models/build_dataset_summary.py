from __future__ import annotations

import argparse

import pandas as pd

from common import make_parser, read_csv, setup_logging, validate_dataset_name, write_csv


def export_dataset_summary(
    dataset: str,
    input_ratings: str,
    input_descriptors: str,
    output_csv: str,
    log_file: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    ratings = read_csv(input_ratings)
    desc = read_csv(input_descriptors)

    m = ratings["user_id"].nunique() if "user_id" in ratings.columns else 0
    n = ratings["item_id"].nunique() if "item_id" in ratings.columns else 0
    num_ratings = len(ratings)
    d = max(desc.shape[1] - 1, 0)
    sparsity = 1.0 - (num_ratings / (m * n)) if m and n else 0.0

    rows = [
        ("num_users", str(m)),
        ("num_items", str(n)),
        ("num_ratings", str(num_ratings)),
        ("rating_scale", "unknown"),
        ("descriptor_source", "from_processed_descriptor_file"),
        ("descriptor_dim", str(d)),
        ("split_ratio", "80/10/10"),
        ("sparsity", f"{sparsity:.6f}"),
    ]
    out = pd.DataFrame(rows, columns=["statistic", "value"])
    write_csv(out, output_csv, schema_name="dataset_summary")
    logger.info("Dataset summary written to %s", output_csv)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Export manuscript dataset summary CSV.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_ratings", required=True)
    p.add_argument("--input_descriptors", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--overwrite", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    export_dataset_summary(**vars(args))


if __name__ == "__main__":
    main()
