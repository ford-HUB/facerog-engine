import threading
from dataclasses import dataclass

import numpy as np

EMPTY_DIMENSION = 512


@dataclass(frozen=True)
class GallerySnapshot:
    """An immutable view of every indexed template.

    embeddings is an (N, D) matrix of L2-normalized vectors; person_ids[i] owns row i.
    """

    embeddings: np.ndarray
    person_ids: list[str]
    embedding_model: str

    @property
    def size(self) -> int:
        return len(self.person_ids)


class InMemoryVectorIndex:
    """NumPy-backed template store. A brute-force scan is fast for thousands of templates.

    Writers rebuild the matrix and swap it in; readers take a snapshot, so the video
    thread can search while the API enrolls without holding a lock during the search.
    """

    def __init__(self, embedding_model: str) -> None:
        self._embedding_model = embedding_model
        self._lock = threading.Lock()
        self._snapshot = GallerySnapshot(
            np.zeros((0, EMPTY_DIMENSION), dtype=np.float32), [], embedding_model
        )

    def snapshot(self) -> GallerySnapshot:
        return self._snapshot

    def add(self, person_id: str, embeddings: list[np.ndarray]) -> None:
        if not embeddings:
            return
        with self._lock:
            current = self._snapshot
            new_rows = np.stack(embeddings).astype(np.float32)
            matrix = new_rows if current.size == 0 else np.vstack([current.embeddings, new_rows])
            person_ids = current.person_ids + [person_id] * len(embeddings)
            self._snapshot = GallerySnapshot(matrix, person_ids, self._embedding_model)

    def remove(self, person_id: str) -> None:
        with self._lock:
            current = self._snapshot
            keep = [i for i, owner in enumerate(current.person_ids) if owner != person_id]
            self._snapshot = GallerySnapshot(
                current.embeddings[keep], [current.person_ids[i] for i in keep], self._embedding_model
            )

    def clear(self) -> None:
        with self._lock:
            self._snapshot = GallerySnapshot(
                np.zeros((0, EMPTY_DIMENSION), dtype=np.float32), [], self._embedding_model
            )
