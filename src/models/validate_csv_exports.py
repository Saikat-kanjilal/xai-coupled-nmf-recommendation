from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import make_parser, read_csv, setup_logging, validate_dataset_name
from schemas import SCHEMAS


DEFAULT_FILES = {
    "dataset_summary": "results/csv/{dataset}_dataset_summary.csv",
    "main_comparison": "results/csv/{dataset}_main_comparison.csv",
    "ablation_results": "results/csv/{dataset}_ablation_results.csv",
    "k_sensitivity": "results/csv/{dataset}_k_sensitivity.csv",
    "alpha_sensitivity": "results/csv/{dataset}_alpha_sensitivity.csv",
    "sparsity_robustness": "results/csv/{dataset}_sparsity_robustness.csv",
    "explanation_metrics": "results/csv/{dataset}_explanation_metrics.csv",
    "factor_keywords": "results/csv/{dataset}_factor_keywords.csv",
    "case_studies": "results/csv/{dataset}_case_studies.csv",
    "significance_tests": "results/csv/{dataset}_significance_tests.csv",
}


def _validate_file(schema_name: str, path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Missing file: {path}"]
    df = pd.read_csv(path)
    expected = list(SCHEMAS[schema_name].keys())
    got = list(df.columns)
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    if missing:
        errors.append(f"{path}: missing columns {missing}")
    if extra:
        errors.append(f"{path}: extra columns {extra}")
    return errors


def validate_csv_exports(
    dataset: str,
    schema_spec: str,
    mapping_sheet: str,
    log_file: str | None = None,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    errors: list[str] = []
    for schema_name, pattern in DEFAULT_FILES.items():
        errors.extend(_validate_file(schema_name, Path(pattern.format(dataset=dataset))))

    if errors:
        for err in errors:
            logger.error(err)
        raise SystemExit(1)
    logger.info("All default CSV exports validated successfully for dataset=%s", dataset)
    logger.info("Note: schema_spec=%s mapping_sheet=%s are accepted for workflow traceability but not parsed by this skeleton.", schema_spec, mapping_sheet)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Validate exported CSV files against locked schemas.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--schema_spec", required=True)
    p.add_argument("--mapping_sheet", required=True)
    p.add_argument("--log_file", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    validate_csv_exports(**vars(args))


if __name__ == "__main__":
    main()
