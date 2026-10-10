from typing import List, Set, Union
import numpy as np


def precision_at_k(recommended_ids: List[Union[int, str]], actual_relevant_ids: Set[Union[int, str]], k: int = 10) -> float:
    """
    Calculates Precision@K: Proportion of recommended items in top-K that are relevant.
    """
    if k <= 0:
        return 0.0

    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0

    relevant_count = sum(1 for item in top_k if item in actual_relevant_ids)
    return float(relevant_count / len(top_k))


def recall_at_k(recommended_ids: List[Union[int, str]], actual_relevant_ids: Set[Union[int, str]], k: int = 10) -> float:
    """
    Calculates Recall@K: Proportion of actual relevant items that were retrieved in top-K.
    """
    if not actual_relevant_ids or k <= 0:
        return 0.0

    top_k = recommended_ids[:k]
    relevant_count = sum(1 for item in top_k if item in actual_relevant_ids)
    return float(relevant_count / len(actual_relevant_ids))


def ndcg_at_k(recommended_ids: List[Union[int, str]], actual_relevant_ids: Set[Union[int, str]], k: int = 10) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at K (NDCG@K).
    Uses binary relevance (1 if item in actual_relevant_ids else 0).
    """
    if not actual_relevant_ids or k <= 0:
        return 0.0

    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0

    # Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in actual_relevant_ids:
            dcg += 1.0 / np.log2(idx + 2)

    # Ideal Discounted Cumulative Gain (IDCG)
    ideal_hits = min(len(actual_relevant_ids), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def evaluate_model(
    recommendations_dict: dict[int, List[Union[int, str]]],
    ground_truth_dict: dict[int, Set[Union[int, str]]],
    k: int = 10
) -> dict[str, float]:
    """
    Evaluates average Precision@K, Recall@K, and NDCG@K across all users.
    """
    precisions, recalls, ndcgs = [], [], []

    for user_id, actual_set in ground_truth_dict.items():
        recs = recommendations_dict.get(user_id, [])
        precisions.append(precision_at_k(recs, actual_set, k=k))
        recalls.append(recall_at_k(recs, actual_set, k=k))
        ndcgs.append(ndcg_at_k(recs, actual_set, k=k))

    return {
        f"Precision@{k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"Recall@{k}": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        f"NDCG@{k}": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
    }
