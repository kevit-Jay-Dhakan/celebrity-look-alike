"""Utilities for comparing embedding vectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import math
from collections import OrderedDict

Vector = Sequence[float]


def _validate_vector(vector: Vector) -> Tuple[float, ...]:
    """Return a tuple version of *vector* ensuring it is one-dimensional."""
    if isinstance(vector, tuple):
        candidate = vector
    else:
        try:
            candidate = tuple(float(component) for component in vector)  # type: ignore[arg-type]
        except TypeError as exc:  # pragma: no cover - defensive guard
            raise TypeError("Embedding vectors must be iterable") from exc
    if not candidate:
        raise ValueError("Embedding vectors must not be empty")
    return candidate


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Return the cosine similarity between vectors ``a`` and ``b``."""
    vec_a = _validate_vector(a)
    vec_b = _validate_vector(b)
    if len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors must have the same dimensionality")
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    return dot / (norm_a * norm_b)


def euclidean_distance(a: Vector, b: Vector) -> float:
    """Return the Euclidean distance between vectors ``a`` and ``b``."""
    vec_a = _validate_vector(a)
    vec_b = _validate_vector(b)
    if len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors must have the same dimensionality")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(vec_a, vec_b)))


@dataclass
class VectorIndex:
    """A lightweight vector database for similarity search."""

    _vectors: "OrderedDict[str, Tuple[float, ...]]" = field(default_factory=OrderedDict)
    _dimension: Optional[int] = None

    def _prepare_vector(self, vector: Vector) -> Tuple[float, ...]:
        candidate = _validate_vector(vector)
        if self._dimension is None:
            self._dimension = len(candidate)
        elif len(candidate) != self._dimension:
            raise ValueError(
                f"Expected vectors with {self._dimension} dimensions, got {len(candidate)}"
            )
        return candidate

    def upsert(self, label: str, embedding: Vector) -> None:
        """Insert or update the vector associated with ``label``."""
        self._vectors[label] = self._prepare_vector(embedding)

    def bulk_upsert(self, items: Iterable[Tuple[str, Vector]]) -> None:
        """Insert or update multiple vectors at once."""
        for label, vector in items:
            self.upsert(label, vector)

    def delete(self, label: str) -> None:
        """Remove ``label`` from the index if present."""
        if label in self._vectors:
            del self._vectors[label]
            if not self._vectors:
                self._dimension = None

    def clear(self) -> None:
        """Remove every vector from the index."""
        self._vectors.clear()
        self._dimension = None

    def iter_embeddings(self) -> Iterator[Tuple[str, Tuple[float, ...]]]:
        """Yield stored embeddings as ``(label, tuple(vector))`` pairs."""
        for label, vector in self._vectors.items():
            yield label, vector

    def _search_cosine(
        self,
        query: Tuple[float, ...],
        *,
        threshold: Optional[float],
        top_k: Optional[int],
    ) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        for label, candidate in self._vectors.items():
            score = cosine_similarity(candidate, query)
            if threshold is not None and score < threshold:
                continue
            results.append((label, score))
        results.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results

    def _search_euclidean(
        self,
        query: Tuple[float, ...],
        *,
        threshold: Optional[float],
        top_k: Optional[int],
    ) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        for label, candidate in self._vectors.items():
            score = euclidean_distance(candidate, query)
            if threshold is not None and score > threshold:
                continue
            results.append((label, score))
        results.sort(key=lambda item: item[1])
        if top_k is not None:
            results = results[:top_k]
        return results

    def search(
        self,
        query_embedding: Vector,
        *,
        metric: str = "cosine",
        threshold: Optional[float] = None,
        top_k: Optional[int] = 1,
    ) -> List[Tuple[str, float]]:
        """Return the closest matches to ``query_embedding``."""
        if not self._vectors:
            return []
        query = self._prepare_vector(query_embedding)
        if metric == "cosine":
            return self._search_cosine(query, threshold=threshold, top_k=top_k)
        if metric == "euclidean":
            return self._search_euclidean(query, threshold=threshold, top_k=top_k)
        raise ValueError(f"Unsupported metric: {metric!r}")


@dataclass
class EmbeddingDatabase:
    """A facade over :class:`VectorIndex` that exposes the legacy API."""

    _index: VectorIndex = field(default_factory=VectorIndex)

    def register_embedding(self, label: str, embedding: Vector) -> None:
        """Register or overwrite an embedding vector for ``label``."""
        self._index.upsert(label, embedding)

    def bulk_register(self, items: Iterable[Tuple[str, Vector]]) -> None:
        """Register a collection of ``(label, embedding)`` pairs."""
        self._index.bulk_upsert(items)

    def iter_embeddings(self) -> Iterator[Tuple[str, Tuple[float, ...]]]:
        """Yield ``(label, embedding)`` pairs stored in the database."""
        return self._index.iter_embeddings()

    def find_best_match(
        self,
        query_embedding: Vector,
        *,
        metric: str = "cosine",
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[str, float]]:
        """Return the label with the closest embedding to ``query_embedding``."""
        matches = self._index.search(
            query_embedding, metric=metric, threshold=threshold, top_k=1
        )
        if not matches:
            return None
        return matches[0]
