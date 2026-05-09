from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from schemas import (
    ALLOWED_DATASETS,
    ALLOWED_DESCRIPTOR_SOURCES,
    ALLOWED_MODEL_NAMES,
    ALLOWED_SPLIT_TYPES,
    SCHEMAS,
)


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    if not value:
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(Path(sys.argv[0]).stem)
    logger.handlers.clear()
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file:
        path = ensure_parent(log_file)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def validate_dataset_name(dataset: str) -> None:
    if dataset not in ALLOWED_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(ALLOWED_DATASETS)}, got: {dataset}")


def validate_descriptor_source(descriptor_source: str) -> None:
    if descriptor_source not in ALLOWED_DESCRIPTOR_SOURCES:
        raise ValueError(
            f"descriptor_source must be one of {sorted(ALLOWED_DESCRIPTOR_SOURCES)}, got: {descriptor_source}"
        )


def validate_split_type(split_type: str) -> None:
    if split_type not in ALLOWED_SPLIT_TYPES:
        raise ValueError(f"split_type must be one of {sorted(ALLOWED_SPLIT_TYPES)}, got: {split_type}")


def validate_model_names(model_names: Iterable[str]) -> None:
    invalid = set(model_names) - ALLOWED_MODEL_NAMES
    if invalid:
        raise ValueError(f"invalid model names: {sorted(invalid)}")


def require_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def empty_schema_df(schema_name: str) -> pd.DataFrame:
    cols = list(SCHEMAS[schema_name].keys())
    return pd.DataFrame(columns=cols)


def write_csv(df: pd.DataFrame, output_csv: str | Path, schema_name: str | None = None) -> None:
    path = ensure_parent(output_csv)
    if schema_name is not None:
        expected = list(SCHEMAS[schema_name].keys())
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        if missing:
            raise ValueError(f"Missing schema columns for {schema_name}: {missing}")
        if extra:
            raise ValueError(f"Unexpected schema columns for {schema_name}: {extra}")
        df = df[expected]
    df.to_csv(path, index=False)


def coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace("-", "_").replace(" ", "_") for c in out.columns]
    return out


def guess_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(f"None of the candidate columns found: {candidates}; available={sorted(cols)}")
    return None


def make_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description)


def write_placeholder_figure(output_fig: str | Path, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    path = ensure_parent(output_fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.4, message, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
