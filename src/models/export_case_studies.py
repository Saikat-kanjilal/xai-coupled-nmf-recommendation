from __future__ import annotations

import argparse
import pandas as pd

from common import make_parser, read_csv, setup_logging, validate_dataset_name, write_csv


def export_case_studies(
    dataset: str,
    input_case_csv: str,
    output_csv: str,
    log_file: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    _ = read_csv(input_case_csv)
    logger.warning("Case-study extraction is TODO. Writing placeholder CSV.")
    out = pd.DataFrame([
        {"dataset_name": dataset, "user_id": "U1", "item_id": "", "item_name": "", "score": 0.0, "dominant_factors": "f2,f5", "explanation_summary": ""},
        {"dataset_name": dataset, "user_id": "U2", "item_id": "", "item_name": "", "score": 0.0, "dominant_factors": "f1,f3", "explanation_summary": ""},
        {"dataset_name": dataset, "user_id": "U3", "item_id": "", "item_name": "", "score": 0.0, "dominant_factors": "f4,f6", "explanation_summary": ""},
    ])
    write_csv(out, output_csv, schema_name="case_studies")
    logger.info("Placeholder case-study CSV written to %s", output_csv)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Export recommendation case studies CSV. Skeleton only.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_case_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--overwrite", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    export_case_studies(**vars(args))


if __name__ == "__main__":
    main()
