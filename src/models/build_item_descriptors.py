from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

import pandas as pd

from common import (
    guess_column,
    make_parser,
    normalize_columns,
    require_file,
    setup_logging,
    validate_dataset_name,
    validate_descriptor_source,
    write_csv,
)


def _clean_token(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_ ]+", " ", str(text).lower())
    return "_".join(text.split())


def _build_genre_df(items: pd.DataFrame) -> pd.DataFrame:
    item_col = guess_column(items, ["item_id", "movie_id", "movieid", "item"])
    genre_col = guess_column(items, ["genres", "genre"], required=False)
    if genre_col is None:
        return pd.DataFrame({"item_id": items[item_col].astype(str)})
    rows = []
    vocab = set()
    for _, row in items[[item_col, genre_col]].fillna("").iterrows():
        item_id = str(row[item_col])
        raw = str(row[genre_col])
        genres = [_clean_token(x) for x in re.split(r"[|,;/]", raw) if str(x).strip()]
        rows.append((item_id, genres))
        vocab.update(genres)
    vocab = sorted(v for v in vocab if v)
    data = []
    for item_id, genres in rows:
        entry = {"item_id": item_id}
        genre_set = set(genres)
        for g in vocab:
            entry[f"genre_{g}"] = 1.0 if g in genre_set else 0.0
        data.append(entry)
    return pd.DataFrame(data)


def _build_tag_df(tags: pd.DataFrame, top_n_tags: int = 200) -> pd.DataFrame:
    item_col = guess_column(tags, ["item_id", "movie_id", "movieid", "item"])
    tag_col = guess_column(tags, ["tag", "tags", "label"])
    grouped: dict[str, list[str]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for _, row in tags[[item_col, tag_col]].fillna("").iterrows():
        item_id = str(row[item_col])
        token = _clean_token(row[tag_col])
        if token:
            grouped[item_id].append(token)
            counts[token] += 1
    vocab = [tok for tok, _ in counts.most_common(top_n_tags)]
    data = []
    for item_id, toks in grouped.items():
        entry = {"item_id": item_id}
        tok_counts = Counter(toks)
        for tok in vocab:
            entry[f"tag_{tok}"] = float(tok_counts.get(tok, 0))
        data.append(entry)
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["item_id"])


def build_item_descriptors(
    dataset: str,
    input_items: str,
    input_tags: str,
    descriptor_source: str,
    output_csv: str,
    top_n_tags: int = 200,
    log_file: str | None = None,
) -> None:
    logger = setup_logging(log_file)
    validate_dataset_name(dataset)
    validate_descriptor_source(descriptor_source)
    items = normalize_columns(pd.read_csv(require_file(input_items)))
    tags = normalize_columns(pd.read_csv(require_file(input_tags))) if require_file(input_tags) else None

    item_col = guess_column(items, ["item_id", "movie_id", "movieid", "item"])
    base = pd.DataFrame({"item_id": items[item_col].astype(str)})
    genre_df = _build_genre_df(items) if descriptor_source in {"genres", "tags_genres"} else pd.DataFrame({"item_id": base["item_id"]})
    tag_df = _build_tag_df(tags, top_n_tags=top_n_tags) if descriptor_source in {"tags", "tags_genres"} else pd.DataFrame({"item_id": base["item_id"]})

    out = base.merge(genre_df, on="item_id", how="left").merge(tag_df, on="item_id", how="left")
    out = out.fillna(0.0)
    write_csv(out, output_csv)
    logger.info("Descriptor matrix written to %s with shape=%s", output_csv, out.shape)


def build_parser() -> argparse.ArgumentParser:
    p = make_parser("Build item descriptor matrix from items and tags.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--input_items", required=True)
    p.add_argument("--input_tags", required=True)
    p.add_argument("--descriptor_source", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--top_n_tags", type=int, default=200)
    p.add_argument("--log_file", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    build_item_descriptors(**vars(args))


if __name__ == "__main__":
    main()
