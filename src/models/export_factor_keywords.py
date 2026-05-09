from __future__ import annotations

import argparse
import pandas as pd

from common import make_parser, read_csv, setup_logging, validate_dataset_name, write_csv


def export_factor_keywords(
    dataset: str,
    input_factor_csv: str,
    output_csv: str,
    log_file: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    df = read_csv(input_factor_csv)
    logger.warning("Factor keyword extraction is TODO. Writing placeholder CSV.")
    out = pd.DataFrame([
        {"dataset_name": dataset, "factor_id": 1, "descriptor_1": "", "descriptor_2": "", "descriptor_3": "", "descriptor_4": "", "descriptor_5": "", "interpretation": ""},
        {"dataset_name": dataset, "factor_id": 2, "descriptor_1": "", "descriptor_2": "", "descriptor_3": "", "descriptor_4": "", "descriptor_5": "", "interpretation": ""},
        {"dataset_name": dataset, "factor_id": 3, "descriptor_1": "", "descriptor_2": "", "descriptor_3": "", "descriptor_4": "", "descriptor_5": "", "interpretation": ""},
        {"dataset_name": dataset, "factor_id": 4, "descriptor_1": "", "descriptor_2": "", "descriptor_3": "", "descriptor_4": "", "descriptor_5": "", "interpretation": ""},
    ])
    write_csv(out, output_csv, schema_name="factor_keywords")
    logger.info("Placeholder factor keywords CSV written to %s", output_csv)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Export representative factor keywords CSV. Skeleton only.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_factor_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--overwrite", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    export_factor_keywords(**vars(args))


if __name__ == "__main__":
    main()
