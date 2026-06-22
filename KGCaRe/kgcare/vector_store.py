from __future__ import annotations

from pathlib import Path
from typing import List

import faiss
import numpy as np

from kgcare.llm import OpenAIEmbeddingClient
from kgcare.schemas import ChunkRecord, VectorMetadata, read_jsonl_models, write_jsonl_models


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype="float32")


class FaissChunkStore:
    def __init__(self, index_path: Path, metadata_path: Path, embedding_client: OpenAIEmbeddingClient) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_client = embedding_client
        self.index: faiss.Index | None = None
        self.metadata: list[VectorMetadata] = []

    def build(self, chunks: list[ChunkRecord], dimension: int, batch_size: int = 128) -> None:
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_client.embed_texts(texts, batch_size=batch_size)
        matrix = np.array(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] != dimension:
            raise ValueError(f"Expected embedding dimension {dimension}, got shape {matrix.shape}")
        matrix = _normalize(matrix)
        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        metadata = [
            VectorMetadata(
                vector_id=i,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                dataset=chunk.dataset,
                title=chunk.title,
                source_path=chunk.source_path,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for i, chunk in enumerate(chunks)
        ]
        write_jsonl_models(self.metadata_path, metadata)
        self.index = index
        self.metadata = metadata

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = [record for record in read_jsonl_models(self.metadata_path, VectorMetadata)]  # type: ignore[list-item]

    def search(self, query: str, top_k: int = 5) -> list[tuple[VectorMetadata, float]]:
        if self.index is None:
            self.load()
        assert self.index is not None
        query_vector = np.array([self.embedding_client.embed_query(query)], dtype="float32")
        query_vector = _normalize(query_vector)
        scores, ids = self.index.search(query_vector, top_k)
        results: list[tuple[VectorMetadata, float]] = []
        for vector_id, score in zip(ids[0], scores[0]):
            if vector_id < 0 or vector_id >= len(self.metadata):
                continue
            results.append((self.metadata[int(vector_id)], float(score)))
        return results

    @property
    def count(self) -> int:
        if self.index is None and self.index_path.exists():
            self.load()
        return int(self.index.ntotal) if self.index is not None else 0
