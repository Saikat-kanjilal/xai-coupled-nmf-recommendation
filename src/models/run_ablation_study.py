from __future__ import annotations

import argparse
import pandas as pd

from common import make_parser, parse_float_list, parse_int_list, read_csv, setup_logging, validate_dataset_name, write_csv


def export_ablation_results(
    dataset: str,
    input_ratings: str,
    input_descriptors: str,
    output_csv: str,
    k_values: str,
    alpha_values: str,
    beta_values: str,
    lambda_values: str,
    seed_list: str,
    log_file: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    _ = read_csv(input_ratings)
    _ = read_csv(input_descriptors)
    k = parse_int_list(k_values)[0] if parse_int_list(k_values) else 20
    lambda_reg = parse_float_list(lambda_values)[0] if parse_float_list(lambda_values) else 0.001
    logger.warning("Ablation execution is TODO. Writing schema-correct placeholder output.")
    out = pd.DataFrame([
        {"dataset_name": dataset, "variant_name": "ratings_only_nmf", "k": k, "alpha": 0.0, "beta": 0.001, "lambda_reg": lambda_reg,
         "rmse": 0.0, "mae": 0.0, "precision_at_10": 0.0, "recall_at_10": 0.0, "ndcg_at_10": 0.0, "fidelity_at_2": 0.0, "mean_coherence": 0.0},
        {"dataset_name": dataset, "variant_name": "no_descriptor_sparsity", "k": k, "alpha": 1.0, "beta": 0.0, "lambda_reg": lambda_reg,
         "rmse": 0.0, "mae": 0.0, "precision_at_10": 0.0, "recall_at_10": 0.0, "ndcg_at_10": 0.0, "fidelity_at_2": 0.0, "mean_coherence": 0.0},
        {"dataset_name": dataset, "variant_name": "coupled_nmf", "k": k, "alpha": 1.0, "beta": 0.001, "lambda_reg": lambda_reg,
         "rmse": 0.0, "mae": 0.0, "precision_at_10": 0.0, "recall_at_10": 0.0, "ndcg_at_10": 0.0, "fidelity_at_2": 0.0, "mean_coherence": 0.0},
    ])
    write_csv(out, output_csv, schema_name="ablation_results")
    logger.info("Placeholder ablation CSV written to %s", output_csv)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Run ablation study and export summary CSV. Skeleton only.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_ratings", required=True)
    p.add_argument("--input_descriptors", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--k_values", required=True)
    p.add_argument("--alpha_values", required=True)
    p.add_argument("--beta_values", required=True)
    p.add_argument("--lambda_values", required=True)
    p.add_argument("--seed_list", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--overwrite", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    export_ablation_results(**vars(args))


if __name__ == "__main__":
    main()
