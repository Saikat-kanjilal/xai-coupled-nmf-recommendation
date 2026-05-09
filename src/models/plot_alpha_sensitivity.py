from __future__ import annotations

import argparse
import pandas as pd

from common import empty_schema_df, make_parser, read_csv, setup_logging, validate_dataset_name, write_csv, write_placeholder_figure


def export_alpha_sensitivity_csv(
    dataset: str,
    input_runs_csv: str,
    output_csv: str,
    output_fig: str,
    log_file: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    _ = read_csv(input_runs_csv)
    logger.warning("Plot aggregation is TODO. Writing schema-correct placeholder CSV and figure.")
    out = empty_schema_df("alpha_sensitivity")
    write_csv(out, output_csv, schema_name="alpha_sensitivity")
    write_placeholder_figure(output_fig, "Sensitivity to coupling parameter alpha", "Placeholder figure. Replace with the real sensitivity plot after experiment aggregation is implemented.")
    logger.info("Placeholder CSV and figure written to %s and %s", output_csv, output_fig)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Export alpha-sensitivity CSV and figure. Skeleton only.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_runs_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_fig", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--overwrite", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    export_alpha_sensitivity_csv(**vars(args))


if __name__ == "__main__":
    main()
