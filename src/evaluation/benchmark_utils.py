from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_seed_list(seed_list: str) -> List[int]:
    return [int(x.strip()) for x in seed_list.split(",") if x.strip()]


def parse_top_k(top_k: str) -> List[int]:
    return [int(x.strip()) for x in top_k.split(",") if x.strip()]


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_log(log_file, message: str) -> None:
    if log_file:
        ensure_parent(log_file)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    print(message)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def load_split_data(
    ratings: pd.DataFrame,
    split_registry: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_seed = split_registry[split_registry["seed"] == seed].copy()

    train_keys = split_seed[split_seed["split"] == "train"][["user_id", "item_id"]]
    val_keys = split_seed[split_seed["split"] == "val"][["user_id", "item_id"]]
    test_keys = split_seed[split_seed["split"] == "test"][["user_id", "item_id"]]

    train = ratings.merge(train_keys, on=["user_id", "item_id"], how="inner")
    val = ratings.merge(val_keys, on=["user_id", "item_id"], how="inner")
    test = ratings.merge(test_keys, on=["user_id", "item_id"], how="inner")

    return train, val, test


def build_id_maps(ratings: pd.DataFrame):
    users = sorted(ratings["user_id"].unique())
    items = sorted(ratings["item_id"].unique())

    user_to_idx = {u: idx for idx, u in enumerate(users)}
    item_to_idx = {i: idx for idx, i in enumerate(items)}

    idx_to_user = {idx: u for u, idx in user_to_idx.items()}
    idx_to_item = {idx: i for i, idx in item_to_idx.items()}

    return user_to_idx, item_to_idx, idx_to_user, idx_to_item


def prediction_metrics(test: pd.DataFrame, pred: np.ndarray):
    y_true = test["rating"].to_numpy(dtype=float)
    y_pred = np.asarray(pred, dtype=float)
    y_pred = np.clip(y_pred, 0.5, 5.0)

    return rmse(y_true, y_pred), mae(y_true, y_pred)


def ranking_metrics_for_scores(
    scores_by_user: Dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_items: List,
    top_k_values: List[int],
    relevance_threshold: float,
) -> Dict[str, float]:
    train_items_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()

    relevant_test = test[test["rating"] >= relevance_threshold]
    relevant_by_user = relevant_test.groupby("user_id")["item_id"].apply(set).to_dict()

    metric_store = {}
    for k in top_k_values:
        metric_store[f"precision_at_{k}"] = []
        metric_store[f"recall_at_{k}"] = []
        metric_store[f"ndcg_at_{k}"] = []

    for user_id, relevant_items in relevant_by_user.items():
        train_seen = train_items_by_user.get(user_id, set())
        candidate_items = [i for i in all_items if i not in train_seen]

        if user_id not in scores_by_user:
            continue

        user_scores = scores_by_user[user_id]
        candidate_scores = [(i, user_scores.get(i, -np.inf)) for i in candidate_items]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        for k in top_k_values:
            top_items = [i for i, _ in candidate_scores[:k]]
            hits = [1 if i in relevant_items else 0 for i in top_items]

            precision = sum(hits) / k
            recall = sum(hits) / len(relevant_items)

            dcg = 0.0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    dcg += 1.0 / np.log2(rank + 1)

            ideal_count = min(len(relevant_items), k)
            idcg = sum(
                1.0 / np.log2(rank + 1)
                for rank in range(1, ideal_count + 1)
            )

            ndcg = dcg / idcg if idcg > 0 else 0.0

            metric_store[f"precision_at_{k}"].append(precision)
            metric_store[f"recall_at_{k}"].append(recall)
            metric_store[f"ndcg_at_{k}"].append(ndcg)

    return {
        key: float(np.mean(values)) if values else 0.0
        for key, values in metric_store.items()
    }
