from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

DatasetName = Literal["conditionalqa", "hotpotqa"]

KGCARE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = KGCARE_ROOT.parent

EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-") or "default"


class Neo4jConfig(BaseModel):
    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = Field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = Field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))


class DatasetPaths(BaseModel):
    docs_path: Path
    questions_path: Path
    kg_docs_path: Optional[Path] = None
    vector_docs_path: Optional[Path] = None


def default_dataset_paths(dataset: DatasetName) -> DatasetPaths:
    if dataset == "conditionalqa":
        return DatasetPaths(
            docs_path=REPO_ROOT / "data" / "docs_dev",
            questions_path=REPO_ROOT / "data" / "dev.json",
        )
    if dataset == "hotpotqa":
        return DatasetPaths(
            docs_path=REPO_ROOT / "data" / "wiki_articles_supported_500",
            kg_docs_path=REPO_ROOT / "data" / "wiki_articles_supported_500",
            vector_docs_path=REPO_ROOT / "data" / "wiki_articles_supported_500",
            questions_path=REPO_ROOT / "data" / "stratified_hotpotqa_500sample_with_tag.json",
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


class IndexConfig(BaseModel):
    dataset: DatasetName
    index_name: str
    kg_model: str = "gpt-4o-2024-08-06"
    qa_model: str = "gpt-3.5-turbo-0125"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    index_dir: Path
    run_root: Path = KGCARE_ROOT / "adapter_runs"
    dataset_paths: DatasetPaths
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    max_triplets_per_chunk: int = 30

    @classmethod
    def create(
        cls,
        dataset: DatasetName,
        index_name: str,
        kg_model: str = "gpt-4o-2024-08-06",
        qa_model: str = "gpt-3.5-turbo-0125",
        embedding_model: str = "text-embedding-3-small",
        index_root: Path | None = None,
        docs_path: Path | None = None,
        questions_path: Path | None = None,
        kg_docs_path: Path | None = None,
        vector_docs_path: Path | None = None,
        neo4j: Neo4jConfig | None = None,
    ) -> "IndexConfig":
        paths = default_dataset_paths(dataset)
        if docs_path is not None:
            paths.docs_path = docs_path
        if questions_path is not None:
            paths.questions_path = questions_path
        if kg_docs_path is not None:
            paths.kg_docs_path = kg_docs_path
        if vector_docs_path is not None:
            paths.vector_docs_path = vector_docs_path

        dim = EMBEDDING_DIMS.get(embedding_model)
        if dim is None:
            raise ValueError(
                f"Unknown embedding dimension for {embedding_model}. "
                "Add it to EMBEDDING_DIMS or pass a supported OpenAI embedding model."
            )
        root = index_root or (KGCARE_ROOT / "indexes")
        return cls(
            dataset=dataset,
            index_name=index_name,
            kg_model=kg_model,
            qa_model=qa_model,
            embedding_model=embedding_model,
            embedding_dimension=dim,
            index_dir=root / dataset / index_name,
            dataset_paths=paths,
            neo4j=neo4j or Neo4jConfig(),
        )

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def triples_path(self) -> Path:
        return self.index_dir / "triples.jsonl"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def vector_metadata_path(self) -> Path:
        return self.index_dir / "vector_metadata.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def run_dir(self, model: str, run_name: str, provider: str = "openai") -> Path:
        return self.run_root / self.dataset / provider / slugify(model) / run_name
