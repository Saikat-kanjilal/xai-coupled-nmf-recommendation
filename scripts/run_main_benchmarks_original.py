import argparse
import os
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

def write_log(log_file: str | None, message: str) -> None:
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

    train_keys = split_seed[split_seed["split"] == "train"][[("user_id"), "item_id"]]
    val_keys = split_seed[split_seed["split"] == "val"][[("user_id"), "item_id"]]
    test_keys = split_seed[split_seed["split"] == "test"][[("user_id"), "item_id"]]

    train = ratings.merge(train_keys, on=[("user_id"), "item_id"], how="inner")
    val = ratings.merge(val_keys, on=[("user_id"), "item_id"], how="inner")
    test = ratings.merge(test_keys, on=[("user_id"), "item_id"], how="inner")

    return train, val, test

def build_id_maps(ratings: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Dict]:
    users = sorted(ratings["user_id"].unique())
    items = sorted(ratings["item_id"].unique())

    user_to_idx = {u: idx for idx, u in enumerate(users)}
    item_to_idx = {i: idx for idx, i in enumerate(items)}

    idx_to_user = {idx: u for u, idx in user_to_idx.items()}
    idx_to_item = {idx: i for i, idx in item_to_idx.items()}

    return user_to_idx, item_to_idx, idx_to_user, idx_to_item

def make_train_matrix(
    train: pd.DataFrame,
    user_to_idx: Dict,
    item_to_idx: Dict,
) -> np.ndarray:
    mat = np.zeros((len(user_to_idx), len(item_to_idx)), dtype=np.float32)

    for row in train.itertuples(index=False):
        u = user_to_idx[row.user_id]
        i = item_to_idx[row.item_id]
        mat[u, i] = float(row.rating)

    return mat

def prediction_metrics(test: pd.DataFrame, pred: np.ndarray) -> Tuple[float, float]:
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
        if len(relevant_items) == 0:
            continue

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
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            metric_store[f"precision_at_{k}"].append(precision)
            metric_store[f"recall_at_{k}"].append(recall)
            metric_store[f"ndcg_at_{k}"].append(ndcg)

    return {
        key: float(np.mean(values)) if values else 0.0
        for key, values in metric_store.items()
    }

def run_item_cf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_items: List,
    top_k_values: List[int],
    relevance_threshold: float,
) -> Dict[str, float]:
    global_mean = train["rating"].mean()
    item_mean = train.groupby("item_id")["rating"].mean().to_dict()

    all_data = pd.concat(
        [
            train[["user_id", "item_id", "rating"]],
            test[["user_id", "item_id", "rating"]],
        ]
    )

    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)

    train_mat = make_train_matrix(train, user_to_idx, item_to_idx)

    item_norms = np.linalg.norm(train_mat, axis=0) + 1e-12
    sim = (train_mat.T @ train_mat) / np.outer(item_norms, item_norms)
    np.fill_diagonal(sim, 0.0)

    preds = []
    scores_by_user = {}

    for user_id, group in train.groupby("user_id"):
        rated_items = group["item_id"].tolist()
        rated_ratings = group["rating"].to_numpy(dtype=float)

        rated_indices = [item_to_idx[i] for i in rated_items if i in item_to_idx]

        user_scores = {}

        for item_id in all_items:
            if item_id not in item_to_idx:
                user_scores[item_id] = item_mean.get(item_id, global_mean)
                continue

            j = item_to_idx[item_id]
            sims = sim[j, rated_indices]

            denom = np.sum(np.abs(sims))

            if denom > 1e-12:
                score = float(np.dot(sims, rated_ratings[:len(sims)]) / denom)
            else:
                score = item_mean.get(item_id, global_mean)

            user_scores[item_id] = score

        scores_by_user[user_id] = user_scores

    for row in test.itertuples(index=False):
        if row.user_id in scores_by_user:
            pred = scores_by_user[row.user_id].get(
                row.item_id,
                item_mean.get(row.item_id, global_mean),
            )
        else:
            pred = item_mean.get(row.item_id, global_mean)

        preds.append(pred)

    out = {}
    out["rmse"], out["mae"] = prediction_metrics(test, np.array(preds))

    out.update(
        ranking_metrics_for_scores(
            scores_by_user=scores_by_user,
            train=train,
            test=test,
            all_items=all_items,
            top_k_values=top_k_values,
            relevance_threshold=relevance_threshold,
        )
    )

    return out

def train_biased_mf(
    train: pd.DataFrame,
    user_to_idx: Dict,
    item_to_idx: Dict,
    k: int = 30,
    epochs: int = 25,
    lr: float = 0.01,
    reg: float = 0.05,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    mu = float(train["rating"].mean())
    bu = np.zeros(n_users, dtype=np.float32)
    bi = np.zeros(n_items, dtype=np.float32)

    P = 0.05 * rng.standard_normal((n_users, k)).astype(np.float32)
    Q = 0.05 * rng.standard_normal((n_items, k)).astype(np.float32)

    triples = [
        (user_to_idx[r.user_id], item_to_idx[r.item_id], float(r.rating))
        for r in train.itertuples(index=False)
        if r.user_id in user_to_idx and r.item_id in item_to_idx
    ]

    for _ in range(epochs):
        rng.shuffle(triples)

        for u, i, rating in triples:
            pred = mu + bu[u] + bi[i] + float(P[u] @ Q[i])
            err = rating - pred

            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

            old_p = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * old_p - reg * Q[i])

    return mu, bu, bi, P, Q

def run_biased_mf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_items: List,
    top_k_values: List[int],
    relevance_threshold: float,
    seed: int,
) -> Dict[str, float]:
    all_data = pd.concat(
        [
            train[["user_id", "item_id", "rating"]],
            test[["user_id", "item_id", "rating"]],
        ]
    )

    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)

    mu, bu, bi, P, Q = train_biased_mf(
        train=train,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        seed=seed,
    )

    preds = []

    for row in test.itertuples(index=False):
        u = user_to_idx.get(row.user_id)
        i = item_to_idx.get(row.item_id)

        if u is None or i is None:
            preds.append(mu)
        else:
            preds.append(mu + bu[u] + bi[i] + float(P[u] @ Q[i]))

    scores_by_user = {}

    for user_id, u in user_to_idx.items():
        scores = mu + bu[u] + bi + (Q @ P[u])
        scores_by_user[user_id] = {
            idx_to_item[i]: float(scores[i])
            for i in range(len(idx_to_item))
        }

    out = {}
    out["rmse"], out["mae"] = prediction_metrics(test, np.array(preds))

    out.update(
        ranking_metrics_for_scores(
            scores_by_user=scores_by_user,
            train=train,
            test=test,
            all_items=all_items,
            top_k_values=top_k_values,
            relevance_threshold=relevance_threshold,
        )
    )

    return out

def train_plain_nmf(
    train: pd.DataFrame,
    user_to_idx: Dict,
    item_to_idx: Dict,
    k: int = 30,
    epochs: int = 35,
    lr: float = 0.005,
    reg: float = 0.03,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    U = rng.random((n_users, k), dtype=np.float32) * 0.1
    V = rng.random((n_items, k), dtype=np.float32) * 0.1

    triples = [
        (user_to_idx[r.user_id], item_to_idx[r.item_id], float(r.rating))
        for r in train.itertuples(index=False)
        if r.user_id in user_to_idx and r.item_id in item_to_idx
    ]

    for _ in range(epochs):
        rng.shuffle(triples)

        for u, i, rating in triples:
            pred = float(U[u] @ V[i])
            err = rating - pred

            old_u = U[u].copy()

            U[u] += lr * (err * V[i] - reg * U[u])
            V[i] += lr * (err * old_u - reg * V[i])

            U[u] = np.maximum(U[u], 1e-8)
            V[i] = np.maximum(V[i], 1e-8)

    return U, V

def run_plain_nmf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_items: List,
    top_k_values: List[int],
    relevance_threshold: float,
    seed: int,
) -> Dict[str, float]:
    all_data = pd.concat(
        [
            train[["user_id", "item_id", "rating"]],
            test[["user_id", "item_id", "rating"]],
        ]
    )

    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)

    U, V = train_plain_nmf(
        train=train,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        seed=seed,
    )

    global_mean = train["rating"].mean()

    preds = []

    for row in test.itertuples(index=False):
        u = user_to_idx.get(row.user_id)
        i = item_to_idx.get(row.item_id)

        if u is None or i is None:
            preds.append(global_mean)
        else:
            preds.append(float(U[u] @ V[i]))

    scores_by_user = {}

    for user_id, u in user_to_idx.items():
        scores = V @ U[u]
        scores_by_user[user_id] = {
            idx_to_item[i]: float(scores[i])
            for i in range(len(idx_to_item))
        }

    out = {}
    out["rmse"], out["mae"] = prediction_metrics(test, np.array(preds))

    out.update(
        ranking_metrics_for_scores(
            scores_by_user=scores_by_user,
            train=train,
            test=test,
            all_items=all_items,
            top_k_values=top_k_values,
            relevance_threshold=relevance_threshold,
        )
    )

    return out

def aggregate_results(run_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    metric_cols = [
        "rmse",
        "mae",
        "precision_at_5",
        "precision_at_10",
        "recall_at_5",
        "recall_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
    ]

    rows = []

    for model_name, group in run_df.groupby("model_name"):
        row = {
            "dataset_name": dataset_name,
            "model_name": model_name,
        }

        for col in metric_cols:
            row[col] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=0)

        rows.append(row)

    return pd.DataFrame(rows)

def export_main_comparison(
    dataset: str,
    input_ratings: str,
    input_descriptors: str,
    seed_list: str,
    descriptor_source: str,
    top_k: str,
    relevance_threshold: float,
    output_csv: str,
    log_file: str | None = None,
    overwrite: str = "false",
):
    del input_descriptors
    del descriptor_source
    del overwrite

    ensure_parent(output_csv)

    ratings = pd.read_csv(input_ratings)
    split_path = f"results/csv/{dataset}_split_registry.csv"

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split registry not found: {split_path}")

    split_registry = pd.read_csv(split_path)

    seeds = parse_seed_list(seed_list)
    top_k_values = parse_top_k(top_k)

    required_topk = {5, 10}
    if not required_topk.issubset(set(top_k_values)):
        raise ValueError("This workflow expects --top_k to include 5 and 10.")

    all_items = sorted(ratings["item_id"].unique())

    run_rows = []

    for seed in seeds:
        write_log(log_file, f"Running seed={seed}")
        train, val, test = load_split_data(ratings, split_registry, seed)

        models = {
            "item_cf": lambda: run_item_cf(
                train=train,
                test=test,
                all_items=all_items,
                top_k_values=top_k_values,
                relevance_threshold=relevance_threshold,
            ),
            "biased_mf": lambda: run_biased_mf(
                train=train,
                test=test,
                all_items=all_items,
                top_k_values=top_k_values,
                relevance_threshold=relevance_threshold,
                seed=seed,
            ),
            "plain_nmf": lambda: run_plain_nmf(
                train=train,
                test=test,
                all_items=all_items,
                top_k_values=top_k_values,
                relevance_threshold=relevance_threshold,
                seed=seed,
            ),
        }

        for model_name, runner in models.items():
            write_log(log_file, f"  Model: {model_name}")
            metrics = runner()

            row = {
                "dataset_name": dataset,
                "model_name": model_name,
                "seed": seed,
            }

            row.update(metrics)
            run_rows.append(row)

    run_df = pd.DataFrame(run_rows)

    run_level_path = f"results/csv/{dataset}_run_level_metrics.csv"
    ensure_parent(run_level_path)
    run_df.to_csv(run_level_path, index=False)

    summary_df = aggregate_results(run_df, dataset)
    summary_df.to_csv(output_csv, index=False)

    write_log(log_file, f"Saved run-level metrics: {run_level_path}")
    write_log(log_file, f"Saved main comparison: {output_csv}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input_ratings", required=True)
    parser.add_argument("--input_descriptors", required=True)
    parser.add_argument("--seed_list", required=True)
    parser.add_argument("--descriptor_source", default="tags_genres")
    parser.add_argument("--top_k", default="5,10")
    parser.add_argument("--relevance_threshold", type=float, default=4.0)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--overwrite", default="false")

    args = parser.parse_args()
    export_main_comparison(**vars(args))


if __name__ == "__main__":
    main()
