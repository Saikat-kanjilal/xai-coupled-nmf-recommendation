import argparse
import os

import numpy as np
import pandas as pd

from benchmark_utils import (
    build_id_maps,
    ensure_parent,
    load_split_data,
    parse_top_k,
    write_log,
)

from run_coupled_nmf import (
    train_coupled_nmf,
    build_descriptor_matrix,
)

def compute_mean_factor_coherence(descriptors, B, feature_cols, top_m=5):
    descriptor_matrix = descriptors[feature_cols].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(descriptor_matrix, axis=0) + 1e-12
    normalized = descriptor_matrix / norms

    factor_scores = []
    for factor_idx in range(B.shape[1]):
        weights = B[:, factor_idx]
        top_indices = np.argsort(weights)[::-1][:top_m]
        pair_scores = []
        for a_pos in range(len(top_indices)):
            for b_pos in range(a_pos + 1, len(top_indices)):
                a = top_indices[a_pos]
                b = top_indices[b_pos]
                sim = float(np.dot(normalized[:, a], normalized[:, b]))
                pair_scores.append(sim)

        if pair_scores:
            factor_scores.append(float(np.mean(pair_scores)))

    if not factor_scores:
        return np.nan

    return float(np.mean(factor_scores))

def compute_mean_factor_coherence(descriptors, B, feature_cols, top_m=5):
    descriptor_matrix = descriptors[feature_cols].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(descriptor_matrix, axis=0) + 1e-12
    normalized = descriptor_matrix / norms

    factor_scores = []
    for factor_idx in range(B.shape[1]):
        weights = B[:, factor_idx]
        top_indices = np.argsort(weights)[::-1][:top_m]
        pair_scores = []
        for a_pos in range(len(top_indices)):
            for b_pos in range(a_pos + 1, len(top_indices)):
                a = top_indices[a_pos]
                b = top_indices[b_pos]
                sim = float(np.dot(normalized[:, a], normalized[:, b]))
                pair_scores.append(sim)

        if pair_scores:
            factor_scores.append(float(np.mean(pair_scores)))

    if not factor_scores:
        return np.nan

    return float(np.mean(factor_scores))

def compute_explanation_instances(
    dataset,
    train,
    descriptors,
    model_objects,
    top_k_recs=10,
    max_users=100,
):
    U = model_objects["U"]
    V = model_objects["V"]
    idx_to_item = model_objects["idx_to_item"]
    user_to_idx = model_objects["user_to_idx"]

    train_items_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()

    rows = []
    selected_users = list(user_to_idx.items())[:max_users]

    for user_id, u in selected_users:
        seen = train_items_by_user.get(user_id, set())
        scores = V @ U[u]
        candidate_pairs = []

        for i_idx, item_id in idx_to_item.items():
            if item_id not in seen:
                candidate_pairs.append((item_id, float(scores[i_idx]), i_idx))

        candidate_pairs.sort(key=lambda x: x[1], reverse=True)

        for rank, (item_id, score, i_idx) in enumerate(candidate_pairs[:top_k_recs], start=1):
            contributions = U[u] * V[i_idx]
            total_contribution = float(np.sum(contributions) + 1e-12)

            sorted_factors = np.argsort(contributions)[::-1]
            top2 = sorted_factors[:2]
            top3 = sorted_factors[:3]

            fidelity_at_2 = float(np.sum(contributions[top2]) / total_contribution)
            fidelity_at_3 = float(np.sum(contributions[top3]) / total_contribution)

            dominant_factors = ",".join([f"f{x + 1}" for x in top3])

            rows.append({
                "dataset_name": dataset,
                "model_name": "coupled_nmf",
                "user_id": user_id,
                "item_id": item_id,
                "rank": rank,
                "score": score,
                "dominant_factors": dominant_factors,
                "fidelity_at_2": fidelity_at_2,
                "fidelity_at_3": fidelity_at_3,
            })

    return pd.DataFrame(rows)

def compute_explanation_instances(
    dataset,
    train,
    descriptors,
    model_objects,
    top_k_recs=10,
    max_users=100,
):
    U = model_objects["U"]
    V = model_objects["V"]
    idx_to_item = model_objects["idx_to_item"]
    user_to_idx = model_objects["user_to_idx"]

    train_items_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()

    rows = []
    selected_users = list(user_to_idx.items())[:max_users]

    for user_id, u in selected_users:
        seen = train_items_by_user.get(user_id, set())
        scores = V @ U[u]
        candidate_pairs = []

        for i_idx, item_id in idx_to_item.items():
            if item_id not in seen:
                candidate_pairs.append((item_id, float(scores[i_idx]), i_idx))

        candidate_pairs.sort(key=lambda x: x[1], reverse=True)

        for rank, (item_id, score, i_idx) in enumerate(candidate_pairs[:top_k_recs], start=1):
            contributions = U[u] * V[i_idx]
            total_contribution = float(np.sum(contributions) + 1e-12)

            sorted_factors = np.argsort(contributions)[::-1]
            top2 = sorted_factors[:2]
            top3 = sorted_factors[:3]

            fidelity_at_2 = float(np.sum(contributions[top2]) / total_contribution)
            fidelity_at_3 = float(np.sum(contributions[top3]) / total_contribution)

            dominant_factors = ",".join([f"f{x + 1}" for x in top3])

            rows.append({
                "dataset_name": dataset,
                "model_name": "coupled_nmf",
                "user_id": user_id,
                "item_id": item_id,
                "rank": rank,
                "score": score,
                "dominant_factors": dominant_factors,
                "fidelity_at_2": fidelity_at_2,
                "fidelity_at_3": fidelity_at_3,
            })

    return pd.DataFrame(rows)

def export_explanation_metrics(
    dataset,
    input_ratings,
    input_descriptors,
    selected_config_csv,
    seed,
    top_k,
    relevance_threshold,
    final_epochs,
    max_users,
    top_m_descriptors,
    output_csv,
    output_instances_csv,
    log_file=None,
    overwrite="false",
):
    del relevance_threshold
    del overwrite

    ratings = pd.read_csv(input_ratings)
    descriptors = pd.read_csv(input_descriptors)
    selected = pd.read_csv(selected_config_csv).iloc[0]

    split_path = f"results/csv/{dataset}_split_registry.csv"
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split registry not found: {split_path}")

    split_registry = pd.read_csv(split_path)
    train, val, test = load_split_data(ratings, split_registry, seed)

    all_data = pd.concat(
        [train[["user_id", "item_id", "rating"]], test[["user_id", "item_id", "rating"]]],
        ignore_index=True
    )
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)

    k = int(selected["k"])
    alpha = float(selected["alpha"])
    beta = float(selected["beta"])
    lambda_reg = float(selected["lambda_reg"])

    write_log(log_file, f"Training coupled_nmf for explanations: k={k}, alpha={alpha}")

    U, V, B, feature_cols = train_coupled_nmf(
        train=train, descriptors=descriptors, user_to_idx=user_to_idx,
        item_to_idx=item_to_idx, k=k, epochs=final_epochs,
        alpha=alpha, beta=beta, lambda_reg=lambda_reg, seed=seed
    )

    model_objects = {
        "U": U, "V": V, "B": B, "feature_cols": feature_cols,
        "user_to_idx": user_to_idx, "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user, "idx_to_item": idx_to_item
    }

    top_k_values = parse_top_k(top_k)
    top_k_recs = max(top_k_values)

    instances = compute_explanation_instances(
        dataset=dataset, train=train, descriptors=descriptors,
        model_objects=model_objects, top_k_recs=top_k_recs, max_users=max_users
    )

    mean_fidelity_at_2 = float(instances["fidelity_at_2"].mean())
    mean_fidelity_at_3 = float(instances["fidelity_at_3"].mean())
    mean_coherence = compute_mean_factor_coherence(
        descriptors, B, feature_cols, top_m=top_m_descriptors
    )

    metrics = pd.DataFrame([{
        "dataset_name": dataset,
        "model_name": "coupled_nmf",
        "fidelity_at_2": mean_fidelity_at_2,
        "fidelity_at_3": mean_fidelity_at_3,
        "mean_coherence": mean_coherence,
        "seed": seed, "k": k, "alpha": alpha
    }])

    ensure_parent(output_csv)
    metrics.to_csv(output_csv, index=False)
    instances.to_csv(output_instances_csv, index=False)

    write_log(log_file, f"Saved metrics to {output_csv}")

def export_explanation_metrics(
    dataset,
    input_ratings,
    input_descriptors,
    selected_config_csv,
    seed,
    top_k,
    relevance_threshold,
    final_epochs,
    max_users,
    top_m_descriptors,
    output_csv,
    output_instances_csv,
    log_file=None,
    overwrite="false",
):
    del relevance_threshold
    del overwrite

    ratings = pd.read_csv(input_ratings)
    descriptors = pd.read_csv(input_descriptors)
    selected = pd.read_csv(selected_config_csv).iloc[0]

    split_path = f"results/csv/{dataset}_split_registry.csv"
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split registry not found: {split_path}")

    split_registry = pd.read_csv(split_path)
    train, val, test = load_split_data(ratings, split_registry, seed)

    all_data = pd.concat(
        [train[["user_id", "item_id", "rating"]], test[["user_id", "item_id", "rating"]]],
        ignore_index=True
    )
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)

    k = int(selected["k"])
    alpha = float(selected["alpha"])
    beta = float(selected["beta"])
    lambda_reg = float(selected["lambda_reg"])

    write_log(log_file, f"Training coupled_nmf for explanations: k={k}, alpha={alpha}")

    U, V, B, feature_cols = train_coupled_nmf(
        train=train, descriptors=descriptors, user_to_idx=user_to_idx,
        item_to_idx=item_to_idx, k=k, epochs=final_epochs,
        alpha=alpha, beta=beta, lambda_reg=lambda_reg, seed=seed
    )

    model_objects = {
        "U": U, "V": V, "B": B, "feature_cols": feature_cols,
        "user_to_idx": user_to_idx, "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user, "idx_to_item": idx_to_item
    }

    top_k_values = parse_top_k(top_k)
    top_k_recs = max(top_k_values)

    instances = compute_explanation_instances(
        dataset=dataset, train=train, descriptors=descriptors,
        model_objects=model_objects, top_k_recs=top_k_recs, max_users=max_users
    )

    mean_fidelity_at_2 = float(instances["fidelity_at_2"].mean())
    mean_fidelity_at_3 = float(instances["fidelity_at_3"].mean())
    mean_coherence = compute_mean_factor_coherence(
        descriptors, B, feature_cols, top_m=top_m_descriptors
    )

    metrics = pd.DataFrame([{
        "dataset_name": dataset,
        "model_name": "coupled_nmf",
        "fidelity_at_2": mean_fidelity_at_2,
        "fidelity_at_3": mean_fidelity_at_3,
        "mean_coherence": mean_coherence,
        "seed": seed, "k": k, "alpha": alpha
    }])

    ensure_parent(output_csv)
    metrics.to_csv(output_csv, index=False)
    instances.to_csv(output_instances_csv, index=False)

    write_log(log_file, f"Saved metrics to {output_csv}")

def export_explanation_metrics(
    dataset,
    input_ratings,
    input_descriptors,
    selected_config_csv,
    seed,
    top_k,
    relevance_threshold,
    final_epochs,
    max_users,
    top_m_descriptors,
    output_csv,
    output_instances_csv,
    log_file=None,
    overwrite="false",
):
    del relevance_threshold
    del overwrite

    ratings = pd.read_csv(input_ratings)
    descriptors = pd.read_csv(input_descriptors)
    selected = pd.read_csv(selected_config_csv).iloc[0]

    split_path = f"results/csv/{dataset}_split_registry.csv"
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split registry not found: {split_path}")

    split_registry = pd.read_csv(split_path)
    train, val, test = load_split_data(ratings, split_registry, seed)

    all_data = pd.concat(
        [train[["user_id", "item_id", "rating"]], test[["user_id", "item_id", "rating"]]],
        ignore_index=True
    )
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)

    k = int(selected["k"])
    alpha = float(selected["alpha"])
    beta = float(selected["beta"])
    lambda_reg = float(selected["lambda_reg"])

    write_log(log_file, f"Training coupled_nmf for explanations: k={k}, alpha={alpha}")

    U, V, B, feature_cols = train_coupled_nmf(
        train=train, descriptors=descriptors, user_to_idx=user_to_idx,
        item_to_idx=item_to_idx, k=k, epochs=final_epochs,
        alpha=alpha, beta=beta, lambda_reg=lambda_reg, seed=seed
    )

    model_objects = {
        "U": U, "V": V, "B": B, "feature_cols": feature_cols,
        "user_to_idx": user_to_idx, "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user, "idx_to_item": idx_to_item
    }

    top_k_values = parse_top_k(top_k)
    top_k_recs = max(top_k_values)

    instances = compute_explanation_instances(
        dataset=dataset, train=train, descriptors=descriptors,
        model_objects=model_objects, top_k_recs=top_k_recs, max_users=max_users
    )

    mean_fidelity_at_2 = float(instances["fidelity_at_2"].mean())
    mean_fidelity_at_3 = float(instances["fidelity_at_3"].mean())
    mean_coherence = compute_mean_factor_coherence(
        descriptors, B, feature_cols, top_m=top_m_descriptors
    )

    metrics = pd.DataFrame([{
        "dataset_name": dataset,
        "model_name": "coupled_nmf",
        "fidelity_at_2": mean_fidelity_at_2,
        "fidelity_at_3": mean_fidelity_at_3,
        "mean_coherence": mean_coherence,
        "seed": seed, "k": k, "alpha": alpha
    }])

    ensure_parent(output_csv)
    metrics.to_csv(output_csv, index=False)
    instances.to_csv(output_instances_csv, index=False)

    write_log(log_file, f"Saved metrics to {output_csv}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input_ratings", required=True)
    parser.add_argument("--input_descriptors", required=True)
    parser.add_argument("--selected_config_csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", default="5,10")
    parser.add_argument("--relevance_threshold", type=float, default=4.0)
    parser.add_argument("--final_epochs", type=int, default=30)
    parser.add_argument("--max_users", type=int, default=100)
    parser.add_argument("--top_m_descriptors", type=int, default=5)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_instances_csv", required=True)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--overwrite", default="false")

    args = parser.parse_args()
    export_explanation_metrics(**vars(args))

if __name__ == "__main__":
    main()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input_ratings", required=True)
    parser.add_argument("--input_descriptors", required=True)
    parser.add_argument("--selected_config_csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", default="5,10")
    parser.add_argument("--relevance_threshold", type=float, default=4.0)
    parser.add_argument("--final_epochs", type=int, default=30)
    parser.add_argument("--max_users", type=int, default=100)
    parser.add_argument("--top_m_descriptors", type=int, default=5)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_instances_csv", required=True)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--overwrite", default="false")

    args = parser.parse_args()
    export_explanation_metrics(**vars(args))

if __name__ == "__main__":
    main()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input_ratings", required=True)
    parser.add_argument("--input_descriptors", required=True)
    parser.add_argument("--selected_config_csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", default="5,10")
    parser.add_argument("--relevance_threshold", type=float, default=4.0)
    parser.add_argument("--final_epochs", type=int, default=30)
    parser.add_argument("--max_users", type=int, default=100)
    parser.add_argument("--top_m_descriptors", type=int, default=5)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_instances_csv", required=True)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--overwrite", default="false")

    args = parser.parse_args()
    export_explanation_metrics(**vars(args))

if __name__ == "__main__":
    main()

