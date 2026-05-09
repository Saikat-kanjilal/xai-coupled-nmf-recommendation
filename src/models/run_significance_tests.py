from __future__ import annotations

import argparse
import pandas as pd

from common import make_parser, read_csv, setup_logging, validate_dataset_name, write_csv


def export_significance_tests(
    dataset: str,
    input_runs_csv: str,
    output_csv: str,
    log_file: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    _ = read_csv(input_runs_csv)
    logger.warning("Significance testing is TODO. Writing placeholder CSV.")
    out = pd.DataFrame([
        {"dataset_name": dataset, "comparison": "coupled_nmf_vs_plain_nmf", "metric_name": "ndcg_at_10", "test_used": "wilcoxon_signed_rank", "p_value": 1.0, "decision": "not_significant"},
        {"dataset_name": dataset, "comparison": "coupled_nmf_vs_als_mf", "metric_name": "ndcg_at_10", "test_used": "wilcoxon_signed_rank", "p_value": 1.0, "decision": "not_significant"},
    ])
    write_csv(out, output_csv, schema_name="significance_tests")
    logger.info("Placeholder significance CSV written to %s", output_csv)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Run significance tests and export summary CSV. Skeleton only.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_runs_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--overwrite", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    export_significance_tests(**vars(args))


if __name__ == "__main__":
    main()
