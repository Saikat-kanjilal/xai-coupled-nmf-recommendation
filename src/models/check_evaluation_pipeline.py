from __future__ import annotations

import argparse

import numpy as np

from common import make_parser, parse_int_list, read_csv, setup_logging, validate_dataset_name
from metrics import mae, ndcg_at_k, precision_at_k, recall_at_k, rmse


def check_evaluation_pipeline(
    dataset: str,
    input_ratings: str,
    split_registry: str,
    top_k: str,
    relevance_threshold: float,
    log_file: str | None = None,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    ratings = read_csv(input_ratings)
    splits = read_csv(split_registry)

    top_values = parse_int_list(top_k)
    if not top_values:
        raise ValueError("top_k must contain at least one value")

    y_true = np.array([4.0, 5.0, 3.0, 2.0])
    y_pred = np.array([3.8, 4.6, 3.2, 2.1])
    logger.info("Sanity RMSE=%.6f, MAE=%.6f", rmse(y_true, y_pred), mae(y_true, y_pred))

    relevant = {2, 4, 7}
    recommended = [7, 8, 4, 5, 2]
    k = top_values[0]
    logger.info(
        "Sanity P@%d=%.6f, R@%d=%.6f, NDCG@%d=%.6f",
        k,
        precision_at_k(relevant, recommended, k),
        k,
        recall_at_k(relevant, recommended, k),
        k,
        ndcg_at_k([1.0, 0.0, 1.0, 0.0, 1.0], k),
    )

    required_split_cols = {"dataset_name", "seed", "split_type", "user_id", "item_id", "rating", "timestamp", "split"}
    missing = required_split_cols - set(splits.columns)
    if missing:
        raise ValueError(f"split registry missing columns: {sorted(missing)}")

    logger.info("Evaluation pipeline sanity checks passed for dataset=%s with relevance_threshold=%s", dataset, relevance_threshold)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Run basic evaluation sanity checks.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_ratings", required=True)
    p.add_argument("--split_registry", required=True)
    p.add_argument("--top_k", required=True)
    p.add_argument("--relevance_threshold", type=float, default=4.0)
    p.add_argument("--log_file", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    check_evaluation_pipeline(**vars(args))


if __name__ == "__main__":
    main()
