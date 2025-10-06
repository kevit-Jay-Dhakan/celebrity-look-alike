import math

import pytest

from celebrity_look_alike import (
    EmbeddingDatabase,
    VectorIndex,
    cosine_similarity,
    euclidean_distance,
)


def test_cosine_similarity_detects_zero_vectors():
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 1.0])


@pytest.mark.parametrize(
    "a, b, expected",
    [([1, 0], [1, 0], 1.0), ([1, 0], [0, 1], 0.0), ([1, 1], [1, 1], 1.0)],
)
def test_cosine_similarity_values(a, b, expected):
    assert pytest.approx(cosine_similarity(a, b)) == expected


def test_euclidean_distance_values():
    assert euclidean_distance([0, 0], [0, 0]) == 0.0
    assert euclidean_distance([0, 0], [3, 4]) == 5.0


def test_find_best_match_euclidean_threshold():
    db = EmbeddingDatabase()
    db.bulk_register(
        [
            ("alice", (0.0, 0.0)),
            ("bob", (1.0, 1.0)),
        ]
    )

    # Distance to ``alice`` is approximately 0.141 which should be accepted by
    # the threshold.  ``bob`` is at a distance of ~1.204 so gets filtered out.
    result = db.find_best_match([0.1, 0.1], metric="euclidean", threshold=0.2)
    assert result == ("alice", pytest.approx(math.sqrt(0.02)))

    # Tighten the threshold so no candidates qualify.
    assert db.find_best_match([0.1, 0.1], metric="euclidean", threshold=0.05) is None


def test_find_best_match_cosine_threshold():
    db = EmbeddingDatabase()
    db.bulk_register(
        [
            ("alice", (1.0, 0.0)),
            ("bob", (0.0, 1.0)),
        ]
    )

    # ``alice`` should win as it has perfect similarity.
    assert db.find_best_match([1.0, 0.0], metric="cosine", threshold=0.5) == ("alice", 1.0)

    # Threshold excludes matches below the specified similarity score.
    assert db.find_best_match([1.0, 0.0], metric="cosine", threshold=1.1) is None


def test_vector_index_top_k_results():
    index = VectorIndex()
    index.bulk_upsert(
        [
            ("alice", (1.0, 0.0)),
            ("bob", (0.5, 0.5)),
            ("carol", (0.0, 1.0)),
        ]
    )

    matches = index.search([0.9, 0.1], metric="cosine", top_k=2)
    assert [label for label, _ in matches] == ["alice", "bob"]
    assert matches[0][1] > matches[1][1]


def test_vector_index_dimension_mismatch():
    index = VectorIndex()
    index.upsert("alice", (0.0, 1.0))
    with pytest.raises(ValueError):
        index.upsert("bob", (1.0, 0.0, 0.0))


def test_vector_index_deletion_and_clear():
    index = VectorIndex()
    index.bulk_upsert([("alice", (0.0, 1.0)), ("bob", (1.0, 0.0))])
    index.delete("alice")
    assert [label for label, _ in index.iter_embeddings()] == ["bob"]

    index.clear()
    assert list(index.iter_embeddings()) == []
    assert index.search([1.0, 0.0]) == []
