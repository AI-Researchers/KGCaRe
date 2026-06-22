from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from hybridrag.indexing import GRAPH_PATH, TRIPLE_STORE_DIR, TRIPLES_PATH, VECTOR_STORE_DIR, normalize_node
from hybridrag.schemas import KGTriple


@dataclass
class RetrievedContext:
    vector_context: str
    graph_context: str
    vector_items: list[dict[str, Any]]
    graph_items: list[dict[str, Any]]


class HybridRetriever:
    def __init__(self, *, index_dir: Path, embeddings) -> None:
        self.index_dir = index_dir
        self.embeddings = embeddings
        self.vector_store = self._load_faiss(index_dir / VECTOR_STORE_DIR)
        self.triple_store = self._load_faiss(index_dir / TRIPLE_STORE_DIR)
        self.graph = self._load_graph(index_dir / GRAPH_PATH)
        self.triples_by_id = self._load_triples(index_dir / TRIPLES_PATH)

    def retrieve(
        self,
        query: str,
        *,
        vector_fetch_k: int = 20,
        vector_context_k: int = 4,
        graph_seed_k: int = 8,
        graph_depth: int = 1,
        graph_context_k: int = 30,
    ) -> RetrievedContext:
        vector_results = self.vector_store.similarity_search_with_score(query, k=vector_fetch_k)
        vector_items = [
            self._doc_to_item(document, score=score, rank=rank)
            for rank, (document, score) in enumerate(vector_results[:vector_context_k], start=1)
        ]

        seed_results = self.triple_store.similarity_search_with_score(query, k=graph_seed_k)
        graph_items = self._expand_seed_triples(seed_results, graph_depth=graph_depth, graph_context_k=graph_context_k)

        return RetrievedContext(
            vector_context=format_vector_context(vector_items),
            graph_context=format_graph_context(graph_items),
            vector_items=vector_items,
            graph_items=graph_items,
        )

    def _load_faiss(self, path: Path):
        from langchain_community.vectorstores import FAISS

        if not (path / "index.faiss").exists():
            raise FileNotFoundError(f"Missing FAISS index: {path}")
        return FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _load_graph(self, path: Path) -> nx.MultiDiGraph:
        if not path.exists():
            raise FileNotFoundError(f"Missing graph: {path}")
        with path.open("rb") as file_obj:
            return pickle.load(file_obj)

    def _load_triples(self, path: Path) -> dict[str, KGTriple]:
        import json

        triples: dict[str, KGTriple] = {}
        with path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                triple = KGTriple.model_validate(row["triple"])
                triple_id = row.get("triple_id") or triple.metadata.get("triple_id")
                if triple_id:
                    triples[triple_id] = triple
        return triples

    def _doc_to_item(self, document, *, score: float, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "score": float(score),
            "text": document.page_content,
            "metadata": dict(document.metadata),
        }

    def _expand_seed_triples(
        self,
        seed_results: list[tuple[Any, float]],
        *,
        graph_depth: int,
        graph_context_k: int,
    ) -> list[dict[str, Any]]:
        ranked_ids: list[str] = []
        scores: dict[str, float] = {}

        for document, score in seed_results:
            metadata = dict(document.metadata)
            triple_id = metadata.get("triple_id", "")
            if triple_id:
                ranked_ids.append(triple_id)
                scores.setdefault(triple_id, float(score))

            for node_key in [metadata.get("head_key", ""), metadata.get("tail_key", "")]:
                for expanded_id in self._expand_node(node_key, depth=graph_depth):
                    ranked_ids.append(expanded_id)
                    scores.setdefault(expanded_id, float(score))

        seen: set[str] = set()
        items: list[dict[str, Any]] = []
        for triple_id in ranked_ids:
            if triple_id in seen:
                continue
            seen.add(triple_id)
            triple = self.triples_by_id.get(triple_id)
            if triple is None:
                continue
            items.append(
                {
                    "rank": len(items) + 1,
                    "score": scores.get(triple_id),
                    "triple_id": triple_id,
                    "head": triple.head,
                    "head_type": triple.head_type,
                    "relation": triple.relation,
                    "tail": triple.tail,
                    "tail_type": triple.tail_type,
                    "evidence": triple.evidence,
                    "metadata": triple.metadata,
                }
            )
            if len(items) >= graph_context_k:
                break
        return items

    def _expand_node(self, node_key: str, *, depth: int) -> list[str]:
        if not node_key:
            return []
        node_key = normalize_node(node_key)
        if node_key not in self.graph:
            return []

        triple_ids: list[str] = []
        frontier = {node_key}
        visited = {node_key}
        for _ in range(max(0, depth) + 1):
            next_frontier: set[str] = set()
            for node in frontier:
                for _, neighbor, _, data in self.graph.out_edges(node, keys=True, data=True):
                    triple_id = data.get("triple_id")
                    if triple_id:
                        triple_ids.append(triple_id)
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
                for neighbor, _, _, data in self.graph.in_edges(node, keys=True, data=True):
                    triple_id = data.get("triple_id")
                    if triple_id:
                        triple_ids.append(triple_id)
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return triple_ids


def format_vector_context(items: list[dict[str, Any]]) -> str:
    if not items:
        return "[no vector context]"
    parts: list[str] = []
    for item in items:
        metadata = item["metadata"]
        parts.append(
            "\n".join(
                [
                    f"[Vector {item['rank']} | doc={metadata.get('title', '')} | chunk={metadata.get('chunk_id', '')}]",
                    item["text"],
                ]
            )
        )
    return "\n\n".join(parts)


def format_graph_context(items: list[dict[str, Any]]) -> str:
    if not items:
        return "[no graph context]"
    parts: list[str] = []
    for item in items:
        metadata = item["metadata"]
        parts.append(
            "\n".join(
                [
                    f"[Triple {item['rank']} | doc={metadata.get('title', '')} | chunk={metadata.get('chunk_id', '')}]",
                    f"({item['head']}:{item['head_type']}) -[{item['relation']}]-> ({item['tail']}:{item['tail_type']})",
                    f"Evidence: {item['evidence']}",
                ]
            )
        )
    return "\n\n".join(parts)
