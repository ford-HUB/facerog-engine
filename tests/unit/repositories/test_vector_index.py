import numpy as np

from app.repositories.vector_index import InMemoryVectorIndex


def test_vector_index_add_and_remove_updates_snapshot() -> None:
    index = InMemoryVectorIndex("model")
    index.add("alice", [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)])
    index.add("bob", [np.zeros(4, dtype=np.float32)])
    assert index.snapshot().size == 3

    index.remove("alice")

    snapshot = index.snapshot()
    assert snapshot.person_ids == ["bob"]
    assert snapshot.embeddings.shape == (1, 4)


def test_vector_index_snapshot_is_unaffected_by_later_writes() -> None:
    index = InMemoryVectorIndex("model")
    index.add("alice", [np.ones(4, dtype=np.float32)])
    before = index.snapshot()
    index.add("bob", [np.ones(4, dtype=np.float32)])
    assert before.size == 1
