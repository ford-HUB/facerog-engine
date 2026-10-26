from dataclasses import dataclass

import numpy as np

from app.ml.embedder import Embedding, l2_normalize


@dataclass(frozen=True)
class Candidate:
    """A possible identity for a face, with cosine similarity in [-1, 1]."""

    person_id: str
    similarity: float


def is_match(similarity: float, match_threshold: float) -> bool:
    """The one place the match threshold is applied."""
    return similarity >= match_threshold


def rank_candidates(
    query: Embedding, gallery: np.ndarray, gallery_person_ids: list[str], top_k: int
) -> list[Candidate]:
    """Rank enrolled people by their best-matching template.

    query is an L2-normalized (D,) vector; gallery is an L2-normalized (N, D) matrix
    whose row i belongs to gallery_person_ids[i]. A person with several templates
    (different photos / angles) is scored by their closest one.
    """
    if gallery.shape[0] == 0:
        return []
    similarities = gallery @ query
    best_per_person: dict[str, float] = {}
    for person_id, similarity in zip(gallery_person_ids, similarities.tolist(), strict=True):
        if similarity > best_per_person.get(person_id, -1.0):
            best_per_person[person_id] = similarity
    ranked = sorted(best_per_person.items(), key=lambda item: item[1], reverse=True)
    return [Candidate(person_id, similarity) for person_id, similarity in ranked[:top_k]]


def compute_similarity(first: Embedding, second: Embedding) -> float:
    """Cosine similarity of two L2-normalized embeddings, in [-1, 1]."""
    return float(np.dot(first, second))


def aggregate_embeddings(embeddings: list[Embedding], weights: list[float]) -> Embedding:
    """Quality-weighted mean of a track's embeddings, re-normalized.

    Averaging over frames smooths out single angled or blurred frames, so a face
    seen from slightly different angles converges to a stable identity vector.
    """
    stacked = np.stack(embeddings)
    weight_array = np.asarray(weights, dtype=np.float32)[:, None]
    return l2_normalize((stacked * weight_array).sum(axis=0))
