import chromadb
from typing import List, Dict

from libs.utils.ml_model.src.config import VECTOR_DB_PATH

_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
_collection = _client.get_or_create_collection(
    name="celeb_embeddings", metadata={"hnsw:space": "cosine"}
)


def embedding_exists(name: str) -> bool:
    result = _collection.get(ids=[name])
    return len(result.get("ids", [])) > 0


def upsert_embedding(name: str, embedding: List[float], metadata: Dict):
    _collection.upsert(ids=[name], embeddings=[embedding], metadatas=[metadata])


def query_similar(embedding: List[float], top_k: int = 5, gender: str | None = None):
    where = {"dominantGender": gender} if gender else {}
    results = _collection.query(
        query_embeddings=[embedding], n_results=top_k, where=where
    )
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return dict(zip(ids, distances))


def list_celeb_names() -> List[str]:
    data = _collection.get()
    return data.get("ids", [])


def delete_embeddings(names: List[str]):
    _collection.delete(ids=names)
