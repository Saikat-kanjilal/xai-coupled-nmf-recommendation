from __future__ import annotations

import argparse
import pandas as pd

from common import make_parser, read_csv, setup_logging, validate_dataset_name, write_csv, write_placeholder_figure


def export_factor_heatmap(
    dataset: str,
    input_factor_csv: str,
    output_csv: str,
    output_fig: str,
    log_file: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    _ = read_csv(input_factor_csv)
    logger.warning("Factor heatmap export is TODO. Writing placeholder matrix CSV and figure.")
    out = pd.DataFrame(columns=["factor_id", "descriptor_name", "weight"])
    write_csv(out, output_csv, schema_name="factor_heatmap_matrix_long")
    write_placeholder_figure(output_fig, "Factor heatmap", "Placeholder figure. Replace with the real factor-by-descriptor visualization.")
    logger.info("Placeholder factor heatmap outputs written to %s and %s", output_csv, output_fig)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Export factor heatmap matrix and figure. Skeleton only.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_factor_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_fig", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--overwrite", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    export_factor_heatmap(**vars(args))


if __name__ == "__main__":
    main()
