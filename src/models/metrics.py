from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(relevant: Iterable[int], recommended: Iterable[int], k: int) -> float:
    rel = set(relevant)
    rec = list(recommended)[:k]
    if k <= 0:
        return 0.0
    return float(sum(1 for x in rec if x in rel) / k)


def recall_at_k(relevant: Iterable[int], recommended: Iterable[int], k: int) -> float:
    rel = set(relevant)
    rec = list(recommended)[:k]
    if not rel:
        return 0.0
    return float(sum(1 for x in rec if x in rel) / len(rel))


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    scores = np.asarray(relevance_scores[:k], dtype=float)
    if scores.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, scores.size + 2))
    dcg = float(np.sum(scores * discounts))
    ideal = np.sort(scores)[::-1]
    idcg = float(np.sum(ideal * discounts))
    if idcg == 0:
        return 0.0
    return dcg / idcg
