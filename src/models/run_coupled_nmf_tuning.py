import argparse
import os
from itertools import product

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

from run_coupled_nmf import (
    train_coupled_nmf,
    aggregate_results,
    export_factor_descriptor_weights,
    export_factor_keywords,
    export_recommendation_traces,
)

def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]

def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def metric_is_lower_better(metric_name):
    return metric_name in {"rmse", "mae"}

def fit_and_evaluate_coupled(
    train, eval_df, descriptors, all_items, top_k_values, relevance_threshold,
    seed, k, alpha, beta, lambda_reg, epochs
):
    all_data = pd.concat([
        train[["user_id", "item_id", "rating"]],
        eval_df[["user_id", "item_id", "rating"]]
    ], ignore_index=True)

    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)

    U, V, B, feature_cols = train_coupled_nmf(
        train=train, descriptors=descriptors, user_to_idx=user_to_idx,
        item_to_idx=item_to_idx, k=k, epochs=epochs, alpha=alpha,
        beta=beta, lambda_reg=lambda_reg, seed=seed
    )

    global_mean = train["rating"].mean()
    preds = []
    for row in eval_df.itertuples(index=False):
        u = user_to_idx.get(row.user_id)
        i = item_to_idx.get(row.item_id)
        if u is not None and i is not None:
            preds.append(float(U[u] @ V[i]))
        else:
            preds.append(global_mean)

    scores_by_user = {}
    for user_id, u_idx in user_to_idx.items():
        scores = V @ U[u_idx]
        scores_by_user[user_id] = {idx_to_item[i]: float(scores[i]) for i in range(len(idx_to_item))}

    metrics = {}
    metrics["rmse"], metrics["mae"] = prediction_metrics(eval_df, np.array(preds))
    metrics.update(ranking_metrics_for_scores(
        scores_by_user=scores_by_user, train=train, test=eval_df,
        all_items=all_items, top_k_values=top_k_values, relevance_threshold=relevance_threshold
    ))

    model_objects = {
        "U": U, "V": V, "B": B, "feature_cols": feature_cols,
        "user_to_idx": user_to_idx, "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user, "idx_to_item": idx_to_item
    }

    return metrics, model_objects

def fit_and_evaluate_coupled(
    train, eval_df, descriptors, all_items, top_k_values, relevance_threshold,
    seed, k, alpha, beta, lambda_reg, epochs
):
    all_data = pd.concat([
        train[["user_id", "item_id", "rating"]],
        eval_df[["user_id", "item_id", "rating"]]
    ], ignore_index=True)

    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(all_data)

    U, V, B, feature_cols = train_coupled_nmf(
        train=train, descriptors=descriptors, user_to_idx=user_to_idx,
        item_to_idx=item_to_idx, k=k, epochs=epochs, alpha=alpha,
        beta=beta, lambda_reg=lambda_reg, seed=seed
    )

    global_mean = train["rating"].mean()
    preds = []
    for row in eval_df.itertuples(index=False):
        u = user_to_idx.get(row.user_id)
        i = item_to_idx.get(row.item_id)
        if u is not None and i is not None:
            preds.append(float(U[u] @ V[i]))
        else:
            preds.append(global_mean)

    scores_by_user = {}
    for user_id, u_idx in user_to_idx.items():
        scores = V @ U[u_idx]
        scores_by_user[user_id] = {idx_to_item[i]: float(scores[i]) for i in range(len(idx_to_item))}

    metrics = {}
    metrics["rmse"], metrics["mae"] = prediction_metrics(eval_df, np.array(preds))
    metrics.update(ranking_metrics_for_scores(
        scores_by_user=scores_by_user, train=train, test=eval_df,
        all_items=all_items, top_k_values=top_k_values, relevance_threshold=relevance_threshold
    ))

    model_objects = {
        "U": U, "V": V, "B": B, "feature_cols": feature_cols,
        "user_to_idx": user_to_idx, "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user, "idx_to_item": idx_to_item
    }

    return metrics, model_objects

def run_validation_grid(
    dataset,
    ratings,
    descriptors,
    split_registry,
    tune_seed,
    top_k_values,
    relevance_threshold,
    k_values,
    alpha_values,
    beta_values,
    lambda_values,
    tune_epochs,
    selection_metric,
    log_file,
):
    train, val, test = load_split_data(
        ratings=ratings,
        split_registry=split_registry,
        seed=tune_seed,
    )

    all_items = sorted(ratings["item_id"].unique())
    rows = []

    grid = list(product(k_values, alpha_values, beta_values, lambda_values))
    write_log(log_file, f"Total tuning configurations: {len(grid)}")

    for idx, (k, alpha, beta, lambda_reg) in enumerate(grid, start=1):
        write_log(
            log_file,
            f"Tuning {idx}/{len(grid)}: k={k}, alpha={alpha}, beta={beta}, lambda={lambda_reg}",
        )

        metrics, _ = fit_and_evaluate_coupled(
            train=train,
            eval_df=val,
            descriptors=descriptors,
            all_items=all_items,
            top_k_values=top_k_values,
            relevance_threshold=relevance_threshold,
            seed=tune_seed,
            k=k,
            alpha=alpha,
            beta=beta,
            lambda_reg=lambda_reg,
            epochs=tune_epochs,
        )

        row = {
            "dataset_name": dataset,
            "seed": tune_seed,
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "lambda_reg": lambda_reg,
            "epochs": tune_epochs,
        }
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)

def run_validation_grid(
    dataset,
    ratings,
    descriptors,
    split_registry,
    tune_seed,
    top_k_values,
    relevance_threshold,
    k_values,
    alpha_values,
    beta_values,
    lambda_values,
    tune_epochs,
    selection_metric,
    log_file,
):
    train, val, test = load_split_data(
        ratings=ratings,
        split_registry=split_registry,
        seed=tune_seed,
    )

    all_items = sorted(ratings["item_id"].unique())
    rows = []

    grid = list(product(k_values, alpha_values, beta_values, lambda_values))
    write_log(log_file, f"Total tuning configurations: {len(grid)}")

    for idx, (k, alpha, beta, lambda_reg) in enumerate(grid, start=1):
        write_log(
            log_file,
            f"Tuning {idx}/{len(grid)}: k={k}, alpha={alpha}, beta={beta}, lambda={lambda_reg}",
        )

        metrics, _ = fit_and_evaluate_coupled(
            train=train,
            eval_df=val,
            descriptors=descriptors,
            all_items=all_items,
            top_k_values=top_k_values,
            relevance_threshold=relevance_threshold,
            seed=tune_seed,
            k=k,
            alpha=alpha,
            beta=beta,
            lambda_reg=lambda_reg,
            epochs=tune_epochs,
        )

        row = {
            "dataset_name": dataset,
            "seed": tune_seed,
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "lambda_reg": lambda_reg,
            "epochs": tune_epochs,
        }
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)

def select_best_config(grid_df, selection_metric):
    if selection_metric not in grid_df.columns:
        raise ValueError(f"Selection metric not found in tuning grid: {selection_metric}")

    candidate_df = grid_df[grid_df["alpha"] > 0].copy()

    if candidate_df.empty:
        raise ValueError("No positive-alpha configurations available for coupled NMF selection.")

    lower_better = metric_is_lower_better(selection_metric)

    if lower_better:
        candidate_df = candidate_df.sort_values(
            by=[selection_metric, "rmse"],
            ascending=[True, True],
        )
    else:
        candidate_df = candidate_df.sort_values(
            by=[selection_metric, "rmse"],
            ascending=[False, True],
        )

    return candidate_df.iloc[0].to_dict()

def select_best_config(grid_df, selection_metric):
    if selection_metric not in grid_df.columns:
        raise ValueError(f"Selection metric not found in tuning grid: {selection_metric}")

    candidate_df = grid_df[grid_df["alpha"] > 0].copy()

    if candidate_df.empty:
        raise ValueError("No positive-alpha configurations available for coupled NMF selection.")

    lower_better = metric_is_lower_better(selection_metric)

    if lower_better:
        candidate_df = candidate_df.sort_values(
            by=[selection_metric, "rmse"],
            ascending=[True, True],
        )
    else:
        candidate_df = candidate_df.sort_values(
            by=[selection_metric, "rmse"],
            ascending=[False, True],
        )

    return candidate_df.iloc[0].to_dict()

def select_best_config(grid_df, selection_metric):
    if selection_metric not in grid_df.columns:
        raise ValueError(f"Selection metric not found in tuning grid: {selection_metric}")

    candidate_df = grid_df[grid_df["alpha"] > 0].copy()

    if candidate_df.empty:
        raise ValueError("No positive-alpha configurations available for coupled NMF selection.")

    lower_better = metric_is_lower_better(selection_metric)

    if lower_better:
        candidate_df = candidate_df.sort_values(
            by=[selection_metric, "rmse"],
            ascending=[True, True],
        )
    else:
        candidate_df = candidate_df.sort_values(
            by=[selection_metric, "rmse"],
            ascending=[False, True],
        )

    return candidate_df.iloc[0].to_dict()

def run_final_selected_config(
    dataset,
    ratings,
    descriptors,
    split_registry,
    seeds,
    top_k_values,
    relevance_threshold,
    best_config,
    final_epochs,
    log_file,
):
    all_items = sorted(ratings["item_id"].unique())
    run_rows = []
    first_model_objects = None
    first_train = None

    k = int(best_config["k"])
    alpha = float(best_config["alpha"])
    beta = float(best_config["beta"])
    lambda_reg = float(best_config["lambda_reg"])

    for seed in seeds:
        write_log(
            log_file,
            f"Final coupled_nmf seed={seed}, k={k}, alpha={alpha}, beta={beta}, lambda={lambda_reg}",
        )

        train, val, test = load_split_data(
            ratings=ratings, split_registry=split_registry, seed=seed
        )

        metrics, model_objects = fit_and_evaluate_coupled(
            train=train,
            eval_df=test,
            descriptors=descriptors,
            all_items=all_items,
            top_k_values=top_k_values,
            relevance_threshold=relevance_threshold,
            seed=seed,
            k=k,
            alpha=alpha,
            beta=beta,
            lambda_reg=lambda_reg,
            epochs=final_epochs,
        )

        row = {
            "dataset_name": dataset,
            "model_name": "coupled_nmf",
            "seed": seed,
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "lambda_reg": lambda_reg,
            "epochs": final_epochs,
        }

        row.update(metrics)
        run_rows.append(row)

        if first_model_objects is None:
            first_model_objects = model_objects
            first_train = train

    return pd.DataFrame(run_rows), first_model_objects, first_train

def run_final_selected_config(
    dataset,
    ratings,
    descriptors,
    split_registry,
    seeds,
    top_k_values,
    relevance_threshold,
    best_config,
    final_epochs,
    log_file,
):
    all_items = sorted(ratings["item_id"].unique())
    run_rows = []
    first_model_objects = None
    first_train = None

    k = int(best_config["k"])
    alpha = float(best_config["alpha"])
    beta = float(best_config["beta"])
    lambda_reg = float(best_config["lambda_reg"])

    for seed in seeds:
        write_log(
            log_file,
            f"Final coupled_nmf seed={seed}, k={k}, alpha={alpha}, beta={beta}, lambda={lambda_reg}",
        )

        train, val, test = load_split_data(
            ratings=ratings, split_registry=split_registry, seed=seed
        )

        metrics, model_objects = fit_and_evaluate_coupled(
            train=train,
            eval_df=test,
            descriptors=descriptors,
            all_items=all_items,
            top_k_values=top_k_values,
            relevance_threshold=relevance_threshold,
            seed=seed,
            k=k,
            alpha=alpha,
            beta=beta,
            lambda_reg=lambda_reg,
            epochs=final_epochs,
        )

        row = {
            "dataset_name": dataset,
            "model_name": "coupled_nmf",
            "seed": seed,
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "lambda_reg": lambda_reg,
            "epochs": final_epochs,
        }

        row.update(metrics)
        run_rows.append(row)

        if first_model_objects is None:
            first_model_objects = model_objects
            first_train = train

    return pd.DataFrame(run_rows), first_model_objects, first_train

def run_final_selected_config(
    dataset,
    ratings,
    descriptors,
    split_registry,
    seeds,
    top_k_values,
    relevance_threshold,
    best_config,
    final_epochs,
    log_file,
):
    all_items = sorted(ratings["item_id"].unique())
    run_rows = []
    first_model_objects = None
    first_train = None

    k = int(best_config["k"])
    alpha = float(best_config["alpha"])
    beta = float(best_config["beta"])
    lambda_reg = float(best_config["lambda_reg"])

    for seed in seeds:
        write_log(
            log_file,
            f"Final coupled_nmf seed={seed}, k={k}, alpha={alpha}, beta={beta}, lambda={lambda_reg}",
        )

        train, val, test = load_split_data(
            ratings=ratings, split_registry=split_registry, seed=seed
        )

        metrics, model_objects = fit_and_evaluate_coupled(
            train=train,
            eval_df=test,
            descriptors=descriptors,
            all_items=all_items,
            top_k_values=top_k_values,
            relevance_threshold=relevance_threshold,
            seed=seed,
            k=k,
            alpha=alpha,
            beta=beta,
            lambda_reg=lambda_reg,
            epochs=final_epochs,
        )

        row = {
            "dataset_name": dataset,
            "model_name": "coupled_nmf",
            "seed": seed,
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "lambda_reg": lambda_reg,
            "epochs": final_epochs,
        }

        row.update(metrics)
        run_rows.append(row)

        if first_model_objects is None:
            first_model_objects = model_objects
            first_train = train

    return pd.DataFrame(run_rows), first_model_objects, first_train

def export_tuned_coupled_nmf(
    dataset,
    input_ratings,
    input_descriptors,
    seed_list,
    descriptor_source,
    top_k,
    relevance_threshold,
    output_csv,
    k_values,
    alpha_values,
    beta_values,
    lambda_values,
    tune_seed,
    tune_epochs,
    final_epochs,
    selection_metric,
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

    k_values = parse_int_list(k_values)
    alpha_values = parse_float_list(alpha_values)
    beta_values = parse_float_list(beta_values)
    lambda_values = parse_float_list(lambda_values)

    grid_df = run_validation_grid(
        dataset=dataset,
        ratings=ratings,
        descriptors=descriptors,
        split_registry=split_registry,
        tune_seed=tune_seed,
        top_k_values=top_k_values,
        relevance_threshold=relevance_threshold,
        k_values=k_values,
        alpha_values=alpha_values,
        beta_values=beta_values,
        lambda_values=lambda_values,
        tune_epochs=tune_epochs,
        selection_metric=selection_metric,
        log_file=log_file,
    )

    tuning_grid_path = f"results/csv/{dataset}_coupled_nmf_tuning_grid.csv"
    ensure_parent(tuning_grid_path)
    grid_df.to_csv(tuning_grid_path, index=False)

    best_config = select_best_config(grid_df, selection_metric)

    selected_config_path = f"results/csv/{dataset}_coupled_nmf_selected_config.csv"
    pd.DataFrame([best_config]).to_csv(selected_config_path, index=False)

    write_log(log_file, f"Saved tuning grid: {tuning_grid_path}")
    write_log(log_file, f"Saved selected config: {selected_config_path}")
    write_log(log_file, f"Selected config: {best_config}")

    new_coupled_df, first_model_objects, first_train = run_final_selected_config(
        dataset=dataset,
        ratings=ratings,
        descriptors=descriptors,
        split_registry=split_registry,
        seeds=seeds,
        top_k_values=top_k_values,
        relevance_threshold=relevance_threshold,
        best_config=best_config,
        final_epochs=final_epochs,
        log_file=log_file,
    )

    run_level_path = f"results/csv/{dataset}_run_level_metrics.csv"

    if os.path.exists(run_level_path):
        old_run_df = pd.read_csv(run_level_path)
        old_run_df = old_run_df[old_run_df["model_name"] != "coupled_nmf"]
        run_df = pd.concat([old_run_df, new_coupled_df], ignore_index=True)
    else:
        run_df = new_coupled_df

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

def export_tuned_coupled_nmf(
    dataset,
    input_ratings,
    input_descriptors,
    seed_list,
    descriptor_source,
    top_k,
    relevance_threshold,
    output_csv,
    k_values,
    alpha_values,
    beta_values,
    lambda_values,
    tune_seed,
    tune_epochs,
    final_epochs,
    selection_metric,
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

    k_values = parse_int_list(k_values)
    alpha_values = parse_float_list(alpha_values)
    beta_values = parse_float_list(beta_values)
    lambda_values = parse_float_list(lambda_values)

    grid_df = run_validation_grid(
        dataset=dataset,
        ratings=ratings,
        descriptors=descriptors,
        split_registry=split_registry,
        tune_seed=tune_seed,
        top_k_values=top_k_values,
        relevance_threshold=relevance_threshold,
        k_values=k_values,
        alpha_values=alpha_values,
        beta_values=beta_values,
        lambda_values=lambda_values,
        tune_epochs=tune_epochs,
        selection_metric=selection_metric,
        log_file=log_file,
    )

    tuning_grid_path = f"results/csv/{dataset}_coupled_nmf_tuning_grid.csv"
    ensure_parent(tuning_grid_path)
    grid_df.to_csv(tuning_grid_path, index=False)

    best_config = select_best_config(grid_df, selection_metric)

    selected_config_path = f"results/csv/{dataset}_coupled_nmf_selected_config.csv"
    pd.DataFrame([best_config]).to_csv(selected_config_path, index=False)

    write_log(log_file, f"Saved tuning grid: {tuning_grid_path}")
    write_log(log_file, f"Saved selected config: {selected_config_path}")
    write_log(log_file, f"Selected config: {best_config}")

    new_coupled_df, first_model_objects, first_train = run_final_selected_config(
        dataset=dataset,
        ratings=ratings,
        descriptors=descriptors,
        split_registry=split_registry,
        seeds=seeds,
        top_k_values=top_k_values,
        relevance_threshold=relevance_threshold,
        best_config=best_config,
        final_epochs=final_epochs,
        log_file=log_file,
    )

    run_level_path = f"results/csv/{dataset}_run_level_metrics.csv"

    if os.path.exists(run_level_path):
        old_run_df = pd.read_csv(run_level_path)
        old_run_df = old_run_df[old_run_df["model_name"] != "coupled_nmf"]
        run_df = pd.concat([old_run_df, new_coupled_df], ignore_index=True)
    else:
        run_df = new_coupled_df

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

    parser.add_argument("--k_values", default="20,30,40")
    parser.add_argument("--alpha_values", default="0.01,0.05,0.1")
    parser.add_argument("--beta_values", default="0,0.0001,0.001")
    parser.add_argument("--lambda_values", default="0.01,0.03")

    parser.add_argument("--tune_seed", type=int, default=42)
    parser.add_argument("--tune_epochs", type=int, default=20)
    parser.add_argument("--final_epochs", type=int, default=30)
    parser.add_argument("--selection_metric", default="ndcg_at_10")

    parser.add_argument("--log_file", default=None)
    parser.add_argument("--overwrite", default="false")

    args = parser.parse_args()
    export_tuned_coupled_nmf(**vars(args))

if __name__ == "__main__":
    main()

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

    parser.add_argument("--k_values", default="20,30,40")
    parser.add_argument("--alpha_values", default="0,0.01,0.05,0.1")
    parser.add_argument("--beta_values", default="0,0.0001")
    parser.add_argument("--lambda_values", default="0.01")

    parser.add_argument("--tune_seed", type=int, default=42)
    parser.add_argument("--tune_epochs", type=int, default=20)
    parser.add_argument("--final_epochs", type=int, default=30)
    parser.add_argument("--selection_metric", default="ndcg_at_10")

    parser.add_argument("--log_file", default=None)
    parser.add_argument("--overwrite", default="false")

    args = parser.parse_args()
    export_tuned_coupled_nmf(**vars(args))

if __name__ == "__main__":
    main()

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

    parser.add_argument("--k_values", default="20,30,40")
    parser.add_argument("--alpha_values", default="0,0.01,0.05,0.1")
    parser.add_argument("--beta_values", default="0,0.0001")
    parser.add_argument("--lambda_values", default="0.01")

    parser.add_argument("--tune_seed", type=int, default=42)
    parser.add_argument("--tune_epochs", type=int, default=20)
    parser.add_argument("--final_epochs", type=int, default=30)
    parser.add_argument("--selection_metric", default="ndcg_at_10")

    parser.add_argument("--log_file", default=None)
    parser.add_argument("--overwrite", default="false")

    args = parser.parse_args()
    export_tuned_coupled_nmf(**vars(args))

if __name__ == "__main__":
    main()

