import argparse
import os

import numpy as np
import pandas as pd

from benchmark_utils import (
    build_id_maps,
    ensure_parent,
    load_split_data,
    parse_seed_list,
    parse_top_k,
    prediction_metrics,
    ranking_metrics_for_scores,
    write_log,
)

def build_descriptor_matrix(descriptors, item_to_idx):
    feature_cols = [
        c for c in descriptors.columns
        if c != 'item_id' and pd.api.types.is_numeric_dtype(descriptors[c])
    ]
    X = np.zeros((len(item_to_idx), len(feature_cols)), dtype=np.float32)
    desc_indexed = descriptors.set_index('item_id')

    for item_id, idx in item_to_idx.items():
        if item_id in desc_indexed.index:
            values = desc_indexed.loc[item_id, feature_cols].to_numpy(dtype=np.float32)
            X[idx, :] = values

    col_max = np.maximum(X.max(axis=0), 1e-8)
    X = X / col_max
    return X, feature_cols

def train_coupled_nmf(
    train, descriptors, user_to_idx, item_to_idx, k=30, epochs=20, 
    lr_rating=0.005, lr_semantic=0.01, alpha=0.1, lambda_reg=0.03, beta=0.001, seed=42
):
    rng = np.random.default_rng(seed)
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    X, feature_cols = build_descriptor_matrix(descriptors, item_to_idx)
    n_features = X.shape[1]

    U = rng.random((n_users, k), dtype=np.float32) * 0.1
    V = rng.random((n_items, k), dtype=np.float32) * 0.1
    B = rng.random((n_features, k), dtype=np.float32) * 0.1

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
            U[u] = np.maximum(U[u] + lr_rating * (err * V[i] - lambda_reg * U[u]), 1e-8)
            V[i] = np.maximum(V[i] + lr_rating * (err * old_u - lambda_reg * V[i]), 1e-8)

        E = X - (V @ B.T)
        grad_V_sem = (-2.0 * alpha / max(n_features, 1)) * (E @ B)
        grad_B_sem = (-2.0 * alpha / max(n_items, 1)) * (E.T @ V)
        V = np.maximum(V - lr_semantic * (grad_V_sem + 2.0 * lambda_reg * V), 1e-8)
        B = np.maximum(B - lr_semantic * (grad_B_sem + 2.0 * lambda_reg * B + beta), 1e-8)

    return U, V, B, feature_cols

def run_coupled_nmf_one_seed(
    train, test, descriptors, all_items, top_k_values, relevance_threshold, seed
):
    all_data = pd.concat(
        [
            train[["user_id", "item_id", "rating"]],
            test[["user_id", "item_id", "rating"]],
        ],
        ignore_index=True,
    )

    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)

    U, V, B, feature_cols = train_coupled_nmf(
        train=train,
        descriptors=descriptors,
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
    for user_id, u_idx in user_to_idx.items():
        scores = V @ U[u_idx]
        scores_by_user[user_id] = {
            idx_to_item[i]: float(scores[i]) for i in range(len(idx_to_item))
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

    model_objects = {
        "U": U,
        "V": V,
        "B": B,
        "feature_cols": feature_cols,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item,
    }

    return out, model_objects

def run_coupled_nmf_one_seed(
    train, test, descriptors, all_items, top_k_values, relevance_threshold, seed
):
    all_data = pd.concat(
        [
            train[["user_id", "item_id", "rating"]],
            test[["user_id", "item_id", "rating"]],
        ],
        ignore_index=True,
    )

    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)

    U, V, B, feature_cols = train_coupled_nmf(
        train=train,
        descriptors=descriptors,
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
    for user_id, u_idx in user_to_idx.items():
        scores = V @ U[u_idx]
        scores_by_user[user_id] = {
            idx_to_item[i]: float(scores[i]) for i in range(len(idx_to_item))
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

    model_objects = {
        "U": U,
        "V": V,
        "B": B,
        "feature_cols": feature_cols,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item,
    }

    return out, model_objects

def aggregate_results(run_df, dataset_name):
    metric_cols = [
        "rmse", "mae",
        "precision_at_5", "precision_at_10",
        "recall_at_5", "recall_at_10",
        "ndcg_at_5", "ndcg_at_10",
    ]
    rows = []
    for model_name, group in run_df.groupby("model_name"):
        row = {"dataset_name": dataset_name, "model_name": model_name}
        for col in metric_cols:
            row[col] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)

def export_factor_descriptor_weights(dataset, model_objects):
    B = model_objects["B"]
    feature_cols = model_objects["feature_cols"]
    rows = []
    for factor_idx in range(B.shape[1]):
        for desc_idx, desc_name in enumerate(feature_cols):
            rows.append({
                "dataset_name": dataset,
                "factor_id": factor_idx + 1,
                "descriptor_name": desc_name,
                "weight": float(B[desc_idx, factor_idx]),
            })
    out_path = f"results/csv/{dataset}_factor_descriptor_weights.csv"
    ensure_parent(out_path)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def export_factor_keywords(dataset, model_objects, top_m=5):
    B = model_objects["B"]
    feature_cols = model_objects["feature_cols"]
    rows = []
    for factor_idx in range(B.shape[1]):
        weights = B[:, factor_idx]
        top_indices = np.argsort(weights)[::-1][:top_m]
        descriptors = [feature_cols[i] for i in top_indices]
        row = {"dataset_name": dataset, "factor_id": factor_idx + 1}
        for j in range(top_m):
            row[f"descriptor_{j + 1}"] = descriptors[j] if j < len(descriptors) else ""
        row["interpretation"] = ""
        rows.append(row)
    out_path = f"results/csv/{dataset}_factor_keywords.csv"
    ensure_parent(out_path)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def export_recommendation_traces(dataset, train, model_objects, top_k=10):
    U = model_objects["U"]
    V = model_objects["V"]
    idx_to_item = model_objects["idx_to_item"]
    user_to_idx = model_objects["user_to_idx"]
    train_items_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()
    rows = []
    for user_id, u in list(user_to_idx.items())[:50]:
        seen = train_items_by_user.get(user_id, set())
        scores = V @ U[u]
        candidate_pairs = []
        for i_idx, item_id in idx_to_item.items():
            if item_id not in seen:
                candidate_pairs.append((item_id, float(scores[i_idx]), i_idx))
        candidate_pairs.sort(key=lambda x: x[1], reverse=True)
        for item_id, score, i_idx in candidate_pairs[:top_k]:
            contrib = U[u] * V[i_idx]
            top_factors = np.argsort(contrib)[::-1][:3] + 1
            rows.append({
                "dataset_name": dataset,
                "user_id": user_id,
                "item_id": item_id,
                "score": score,
                "dominant_factors": ",".join([f"f{x}" for x in top_factors]),
            })
    out_path = f"results/csv/{dataset}_recommendation_traces.csv"
    ensure_parent(out_path)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def aggregate_results(run_df, dataset_name):
    metric_cols = [
        "rmse", "mae",
        "precision_at_5", "precision_at_10",
        "recall_at_5", "recall_at_10",
        "ndcg_at_5", "ndcg_at_10",
    ]
    rows = []
    for model_name, group in run_df.groupby("model_name"):
        row = {"dataset_name": dataset_name, "model_name": model_name}
        for col in metric_cols:
            row[col] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)

def export_factor_descriptor_weights(dataset, model_objects):
    B = model_objects["B"]
    feature_cols = model_objects["feature_cols"]
    rows = []
    for factor_idx in range(B.shape[1]):
        for desc_idx, desc_name in enumerate(feature_cols):
            rows.append({
                "dataset_name": dataset,
                "factor_id": factor_idx + 1,
                "descriptor_name": desc_name,
                "weight": float(B[desc_idx, factor_idx]),
            })
    out_path = f"results/csv/{dataset}_factor_descriptor_weights.csv"
    ensure_parent(out_path)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def export_factor_keywords(dataset, model_objects, top_m=5):
    B = model_objects["B"]
    feature_cols = model_objects["feature_cols"]
    rows = []
    for factor_idx in range(B.shape[1]):
        weights = B[:, factor_idx]
        top_indices = np.argsort(weights)[::-1][:top_m]
        descriptors = [feature_cols[i] for i in top_indices]
        row = {"dataset_name": dataset, "factor_id": factor_idx + 1}
        for j in range(top_m):
            row[f"descriptor_{j + 1}"] = descriptors[j] if j < len(descriptors) else ""
        row["interpretation"] = ""
        rows.append(row)
    out_path = f"results/csv/{dataset}_factor_keywords.csv"
    ensure_parent(out_path)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def export_recommendation_traces(dataset, train, model_objects, top_k=10):
    U = model_objects["U"]
    V = model_objects["V"]
    idx_to_item = model_objects["idx_to_item"]
    user_to_idx = model_objects["user_to_idx"]
    train_items_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()
    rows = []
    for user_id, u in list(user_to_idx.items())[:50]:
        seen = train_items_by_user.get(user_id, set())
        scores = V @ U[u]
        candidate_pairs = []
        for i_idx, item_id in idx_to_item.items():
            if item_id not in seen:
                candidate_pairs.append((item_id, float(scores[i_idx]), i_idx))
        candidate_pairs.sort(key=lambda x: x[1], reverse=True)
        for item_id, score, i_idx in candidate_pairs[:top_k]:
            contrib = U[u] * V[i_idx]
            top_factors = np.argsort(contrib)[::-1][:3] + 1
            rows.append({
                "dataset_name": dataset,
                "user_id": user_id,
                "item_id": item_id,
                "score": score,
                "dominant_factors": ",".join([f"f{x}" for x in top_factors]),
            })
    out_path = f"results/csv/{dataset}_recommendation_traces.csv"
    ensure_parent(out_path)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def export_coupled_nmf(
    dataset,
    input_ratings,
    input_descriptors,
    seed_list,
    descriptor_source,
    top_k,
    relevance_threshold,
    output_csv,
    log_file=None,
    overwrite="false",
):
    del descriptor_source
    del overwrite

    ratings = pd.read_csv(input_ratings)
    descriptors = pd.read_csv(input_descriptors)
    split_path = f"results/csv/{dataset}_split_registry.csv"

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split registry not found: {split_path}")

    split_registry = pd.read_csv(split_path)
    seeds = parse_seed_list(seed_list)
    top_k_values = parse_top_k(top_k)
    all_items = sorted(ratings["item_id"].unique())

    run_rows = []
    first_model_objects = None
    first_train = None

    for seed in seeds:
        write_log(log_file, f"Running coupled_nmf seed={seed}")
        train, val, test = load_split_data(ratings, split_registry, seed)
        metrics, model_objects = run_coupled_nmf_one_seed(
            train, test, descriptors, all_items, top_k_values, relevance_threshold, seed
        )
        row = {"dataset_name": dataset, "model_name": "coupled_nmf", "seed": seed}
        row.update(metrics)
        run_rows.append(row)

        if first_model_objects is None:
            first_model_objects = model_objects
            first_train = train

    new_run_df = pd.DataFrame(run_rows)
    run_level_path = f"results/csv/{dataset}_run_level_metrics.csv"

    if os.path.exists(run_level_path):
        old_run_df = pd.read_csv(run_level_path)
        old_run_df = old_run_df[old_run_df["model_name"] != "coupled_nmf"]
        run_df = pd.concat([old_run_df, new_run_df], ignore_index=True)
    else:
        run_df = new_run_df

    ensure_parent(run_level_path)
    run_df.to_csv(run_level_path, index=False)

    summary_df = aggregate_results(run_df, dataset)
    ensure_parent(output_csv)
    summary_df.to_csv(output_csv, index=False)

    if first_model_objects is not None:
        p1 = export_factor_descriptor_weights(dataset, first_model_objects)
        p2 = export_factor_keywords(dataset, first_model_objects)
        p3 = export_recommendation_traces(dataset, first_train, first_model_objects)
        write_log(log_file, f"Saved: {p1}")
        write_log(log_file, f"Saved: {p2}")
        write_log(log_file, f"Saved: {p3}")

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
    export_coupled_nmf(**vars(args))

if __name__ == "__main__":
    main()

def export_coupled_nmf(
    dataset,
    input_ratings,
    input_descriptors,
    seed_list,
    descriptor_source,
    top_k,
    relevance_threshold,
    output_csv,
    log_file=None,
    overwrite="false",
):
    del descriptor_source
    del overwrite

    ratings = pd.read_csv(input_ratings)
    descriptors = pd.read_csv(input_descriptors)
    split_path = f"results/csv/{dataset}_split_registry.csv"

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split registry not found: {split_path}")

    split_registry = pd.read_csv(split_path)
    seeds = parse_seed_list(seed_list)
    top_k_values = parse_top_k(top_k)
    all_items = sorted(ratings["item_id"].unique())

    run_rows = []
    first_model_objects = None
    first_train = None

    for seed in seeds:
        write_log(log_file, f"Running coupled_nmf seed={seed}")
        train, val, test = load_split_data(ratings, split_registry, seed)
        metrics, model_objects = run_coupled_nmf_one_seed(
            train, test, descriptors, all_items, top_k_values, relevance_threshold, seed
        )
        row = {"dataset_name": dataset, "model_name": "coupled_nmf", "seed": seed}
        row.update(metrics)
        run_rows.append(row)

        if first_model_objects is None:
            first_model_objects = model_objects
            first_train = train

    new_run_df = pd.DataFrame(run_rows)
    run_level_path = f"results/csv/{dataset}_run_level_metrics.csv"

    if os.path.exists(run_level_path):
        old_run_df = pd.read_csv(run_level_path)
        old_run_df = old_run_df[old_run_df["model_name"] != "coupled_nmf"]
        run_df = pd.concat([old_run_df, new_run_df], ignore_index=True)
    else:
        run_df = new_run_df

    ensure_parent(run_level_path)
    run_df.to_csv(run_level_path, index=False)

    summary_df = aggregate_results(run_df, dataset)
    ensure_parent(output_csv)
    summary_df.to_csv(output_csv, index=False)

    if first_model_objects is not None:
        p1 = export_factor_descriptor_weights(dataset, first_model_objects)
        p2 = export_factor_keywords(dataset, first_model_objects)
        p3 = export_recommendation_traces(dataset, first_train, first_model_objects)
        write_log(log_file, f"Saved: {p1}")
        write_log(log_file, f"Saved: {p2}")
        write_log(log_file, f"Saved: {p3}")

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
    export_coupled_nmf(**vars(args))

if __name__ == "__main__":
    main()

