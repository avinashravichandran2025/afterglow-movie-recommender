import pytest
from afterglow import evaluation


def test_precision_at_k():
    recs = [101, 102, 103, 104, 105]
    actual = {101, 103, 109}
    
    # Precision@5 = 2 relevant / 5 recommended = 0.4
    assert evaluation.precision_at_k(recs, actual, k=5) == 0.4
    # Precision@2 = 1 relevant / 2 recommended = 0.5
    assert evaluation.precision_at_k(recs, actual, k=2) == 0.5


def test_recall_at_k():
    recs = [101, 102, 103, 104, 105]
    actual = {101, 103, 109}

    # Recall@5 = 2 retrieved / 3 total actual = 0.6667
    rec = evaluation.recall_at_k(recs, actual, k=5)
    assert round(rec, 4) == 0.6667


def test_ndcg_at_k():
    recs = [101, 102, 103]
    actual = {101, 102}

    # Ideal ranking puts 101, 102 first -> NDCG should be 1.0
    ndcg = evaluation.ndcg_at_k(recs, actual, k=2)
    assert ndcg == 1.0


def test_evaluate_model():
    recs = {
        1: [101, 102, 103],
        2: [201, 202, 203]
    }
    ground_truth = {
        1: {101, 103},
        2: {202}
    }
    metrics = evaluation.evaluate_model(recs, ground_truth, k=3)
    assert "Precision@3" in metrics
    assert "Recall@3" in metrics
    assert "NDCG@3" in metrics
    assert metrics["Precision@3"] > 0.0
