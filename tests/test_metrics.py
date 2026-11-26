import math

import pytest

from temporalproof.evaluation.metrics import ndcg_at_k, recall_at_k


def test_missing_gold_changes_normalization_even_when_first_hit_is_perfect():
    assert ndcg_at_k([10, 99], {10, 11}, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert recall_at_k([10, 99], {10, 11}, 2) == 0.5


def test_all_relevant_documents_at_top_is_perfect():
    assert ndcg_at_k([11, 10, 99], {10, 11}, 2) == 1.0
    assert recall_at_k([11, 10, 99], {10, 11}, 2) == 1.0


@pytest.mark.parametrize("metric", [ndcg_at_k, recall_at_k])
def test_empty_ranking_and_empty_gold_are_zero(metric):
    assert metric([], {10}, 10) == 0.0
    assert metric([10], set(), 10) == 0.0
    assert metric([99], {10}, 10) == 0.0


def test_short_ranking_does_not_shrink_ideal_denominator():
    assert ndcg_at_k([10], {10, 11}, 10) == pytest.approx(0.6131471927654584)


def test_cutoff_excludes_hits_after_boundary():
    assert ndcg_at_k([99, 10], {10}, 1) == 0.0
    assert recall_at_k([99, 10], {10}, 1) == 0.0


@pytest.mark.parametrize("metric", [ndcg_at_k, recall_at_k])
def test_duplicate_hits_are_rejected(metric):
    with pytest.raises(ValueError, match="unique"):
        metric([10, 10], {10}, 10)


@pytest.mark.parametrize("metric", [ndcg_at_k, recall_at_k])
@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_cutoffs_fail(metric, k):
    with pytest.raises(ValueError, match="positive integer"):
        metric([], set(), k)
