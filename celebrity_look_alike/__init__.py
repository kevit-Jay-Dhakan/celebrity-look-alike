"""Core API for the celebrity look-alike matching utilities."""

from .matcher import EmbeddingDatabase, VectorIndex, cosine_similarity, euclidean_distance

__all__ = ["EmbeddingDatabase", "VectorIndex", "cosine_similarity", "euclidean_distance"]
