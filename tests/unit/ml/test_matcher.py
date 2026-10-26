import numpy as np

from app.ml.embedder import l2_normalize
from app.ml.matcher import aggregate_embeddings, compute_similarity, is_match, rank_candidates


def test_rank_candidates_scores_person_by_best_template() -> None:
    query = l2_normalize(np.array([1.0, 0.0, 0.0]))
    gallery = l2_normalize(np.array([[0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.5, 0.5, 0.0]]))
    candidates = rank_candidates(query, gallery, ["alice", "alice", "bob"], top_k=5)
    assert [c.person_id for c in candidates] == ["alice", "bob"]
    assert candidates[0].similarity > 0.99


def test_rank_candidates_empty_gallery_returns_nothing() -> None:
    query = l2_normalize(np.array([1.0, 0.0]))
    assert rank_candidates(query, np.zeros((0, 2), dtype=np.float32), [], top_k=3) == []


def test_is_match_threshold_is_inclusive() -> None:
    assert is_match(0.4, 0.4)
    assert not is_match(0.39, 0.4)


def test_aggregate_embeddings_weights_toward_higher_quality_frames() -> None:
    frontal = l2_normalize(np.array([1.0, 0.0]))
    angled = l2_normalize(np.array([0.0, 1.0]))
    aggregated = aggregate_embeddings([frontal, angled], weights=[0.9, 0.1])
    assert compute_similarity(aggregated, frontal) > compute_similarity(aggregated, angled)
    assert np.isclose(np.linalg.norm(aggregated), 1.0)
