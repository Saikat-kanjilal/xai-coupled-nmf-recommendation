from __future__ import annotations

import argparse
import pandas as pd

from common import make_parser, read_csv, setup_logging, validate_dataset_name, write_csv


def export_sparsity_robustness(
    dataset: str,
    input_ratings: str,
    input_descriptors: str,
    seed_list: str,
    output_csv: str,
    log_file: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    _ = read_csv(input_ratings)
    _ = read_csv(input_descriptors)
    logger.warning("Sparsity analysis is TODO. Writing placeholder robustness CSV.")
    settings = ["full_density", "density_80", "density_60", "density_40"]
    rows = []
    for setting in settings:
        for model in ["plain_nmf", "coupled_nmf"]:
            rows.append({
                "dataset_name": dataset,
                "setting_name": setting,
                "model_name": model,
                "rmse": 0.0,
                "mae": 0.0,
                "precision_at_10": 0.0,
                "recall_at_10": 0.0,
                "ndcg_at_10": 0.0,
            })
    out = pd.DataFrame(rows)
    write_csv(out, output_csv, schema_name="sparsity_robustness")
    logger.info("Placeholder robustness CSV written to %s", output_csv)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Run sparsity robustness analysis. Skeleton only.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_ratings", required=True)
    p.add_argument("--input_descriptors", required=True)
    p.add_argument("--seed_list", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--overwrite", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    export_sparsity_robustness(**vars(args))


if __name__ == "__main__":
    main()
