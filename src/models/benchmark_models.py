from typing import Dict, List

import numpy as np
import pandas as pd

from benchmark_utils import (
    build_id_maps,
    prediction_metrics,
    ranking_metrics_for_scores,
)

def run_item_mean(
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_items: List,
    top_k_values: List[int],
    relevance_threshold: float,
) -> Dict[str, float]:
    global_mean = train["rating"].mean()
    item_mean = train.groupby("item_id")["rating"].mean().to_dict()

    preds = [
        item_mean.get(row.item_id, global_mean)
        for row in test.itertuples(index=False)
    ]

    scores_by_user = {}
    users = sorted(train["user_id"].unique())

    for user_id in users:
        scores_by_user[user_id] = {
            item_id: item_mean.get(item_id, global_mean)
            for item_id in all_items
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

def run_item_mean(
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_items: List,
    top_k_values: List[int],
    relevance_threshold: float,
) -> Dict[str, float]:
    global_mean = train["rating"].mean()
    item_mean = train.groupby("item_id")["rating"].mean().to_dict()

    preds = [
        item_mean.get(row.item_id, global_mean)
        for row in test.itertuples(index=False)
    ]

    scores_by_user = {}
    users = sorted(train["user_id"].unique())

    for user_id in users:
        scores_by_user[user_id] = {
            item_id: item_mean.get(item_id, global_mean)
            for item_id in all_items
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

def train_biased_mf(
    train: pd.DataFrame,
    user_to_idx: Dict,
    item_to_idx: Dict,
    k: int = 30,
    epochs: int = 20,
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

def train_biased_mf(
    train: pd.DataFrame,
    user_to_idx: Dict,
    item_to_idx: Dict,
    k: int = 30,
    epochs: int = 20,
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
    all_data = pd.concat([
        train[["user_id", "item_id", "rating"]],
        test[["user_id", "item_id", "rating"]],
    ], ignore_index=True)
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)
    mu, bu, bi, P, Q = train_biased_mf(train, user_to_idx, item_to_idx, seed=seed)
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
        scores_by_user[user_id] = {idx_to_item[i]: float(scores[i]) for i in range(len(idx_to_item))}
    out = {}
    out["rmse"], out["mae"] = prediction_metrics(test, np.array(preds))
    out.update(ranking_metrics_for_scores(
        scores_by_user=scores_by_user, train=train, test=test,
        all_items=all_items, top_k_values=top_k_values, relevance_threshold=relevance_threshold))
    return out

def run_biased_mf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_items: List,
    top_k_values: List[int],
    relevance_threshold: float,
    seed: int,
) -> Dict[str, float]:
    all_data = pd.concat([
        train[["user_id", "item_id", "rating"]],
        test[["user_id", "item_id", "rating"]],
    ], ignore_index=True)
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)
    mu, bu, bi, P, Q = train_biased_mf(train, user_to_idx, item_to_idx, seed=seed)
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
        scores_by_user[user_id] = {idx_to_item[i]: float(scores[i]) for i in range(len(idx_to_item))}
    out = {}
    out["rmse"], out["mae"] = prediction_metrics(test, np.array(preds))
    out.update(ranking_metrics_for_scores(
        scores_by_user=scores_by_user, train=train, test=test,
        all_items=all_items, top_k_values=top_k_values, relevance_threshold=relevance_threshold))
    return out

def train_plain_nmf(
    train: pd.DataFrame,
    user_to_idx: Dict,
    item_to_idx: Dict,
    k: int = 30,
    epochs: int = 30,
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
    all_data = pd.concat([
        train[["user_id", "item_id", "rating"]],
        test[["user_id", "item_id", "rating"]],
    ], ignore_index=True)
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)
    U, V = train_plain_nmf(train, user_to_idx, item_to_idx, seed=seed)
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
        scores_by_user[user_id] = {idx_to_item[i]: float(scores[i]) for i in range(len(idx_to_item))}
    out = {}
    out["rmse"], out["mae"] = prediction_metrics(test, np.array(preds))
    out.update(ranking_metrics_for_scores(
        scores_by_user=scores_by_user, train=train, test=test,
        all_items=all_items, top_k_values=top_k_values, relevance_threshold=relevance_threshold))
    return out

def run_plain_nmf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_items: List,
    top_k_values: List[int],
    relevance_threshold: float,
    seed: int,
) -> Dict[str, float]:
    all_data = pd.concat([
        train[["user_id", "item_id", "rating"]],
        test[["user_id", "item_id", "rating"]],
    ], ignore_index=True)
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)
    U, V = train_plain_nmf(train, user_to_idx, item_to_idx, seed=seed)
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
        scores_by_user[user_id] = {idx_to_item[i]: float(scores[i]) for i in range(len(idx_to_item))}
    out = {}
    out["rmse"], out["mae"] = prediction_metrics(test, np.array(preds))
    out.update(ranking_metrics_for_scores(
        scores_by_user=scores_by_user, train=train, test=test,
        all_items=all_items, top_k_values=top_k_values, relevance_threshold=relevance_threshold))
    return out

