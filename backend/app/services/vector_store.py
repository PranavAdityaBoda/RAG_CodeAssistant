"""
Chroma wrapper. Nothing else in the app imports chromadb directly.

Embeddings run locally via sentence-transformers with no API calls.
"""
import logging
import sys

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from app.core.config import settings
from app.core.logging import get_logger
from app.services.chunker import Chunk

logger = get_logger(__name__)

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

_COLLECTION_NAME = "code_chunks"
_client: chromadb.ClientAPI | None = None
_embedding_fn = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


class _FakeEmbeddingFn:
    """Hash-based stand-in for testing without downloading the real model."""
    _DIMS = 16

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in input]

    def _embed(self, text: str) -> list[float]:
        return [float((hash((text, i)) % 1000) / 1000.0) for i in range(self._DIMS)]

    def name(self) -> str:
        return "fake"


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        if settings.use_fake_embeddings:
            logger.warning("Fake embeddings on, for testing only.")
            _embedding_fn = _FakeEmbeddingFn()
        else:
            logger.info("Loading embedding model: %s", settings.embedding_model_name)
            _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model_name
            )
    return _embedding_fn


def get_collection(job_id: str):
    """One collection per job keeps repos isolated from each other."""
    client = _get_client()
    return client.get_or_create_collection(
        name=f"{_COLLECTION_NAME}_{job_id}",
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(job_id: str, chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    collection = get_collection(job_id)
    batch_size = 64
    stored = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "file_path":   c.file_path,
                    "language":    c.language,
                    "chunk_type":  c.chunk_type,
                    "symbol_name": c.symbol_name or "",
                    "start_line":  c.start_line,
                    "end_line":    c.end_line,
                }
                for c in batch
            ],
        )
        stored += len(batch)
    logger.info("Stored %d chunks for job %s", stored, job_id)
    return stored


def query_chunks(job_id: str, query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection(job_id)
    results = collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for doc, meta, dist in zip(
        (results.get("documents") or [[]])[0],
        (results.get("metadatas") or [[]])[0],
        (results.get("distances") or [[]])[0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits
