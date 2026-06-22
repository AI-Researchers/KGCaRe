#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss

from kgcare.config import IndexConfig, Neo4jConfig
from kgcare.graph_store import Neo4jTripleStore


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a KGCaRe index.")
    parser.add_argument("--dataset", choices=["conditionalqa", "hotpotqa"], required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--neo4j-uri")
    parser.add_argument("--neo4j-username")
    parser.add_argument("--neo4j-password")
    parser.add_argument("--neo4j-database")
    parser.add_argument("--sample-triples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    neo4j = Neo4jConfig()
    if args.neo4j_uri:
        neo4j.uri = args.neo4j_uri
    if args.neo4j_username:
        neo4j.username = args.neo4j_username
    if args.neo4j_password:
        neo4j.password = args.neo4j_password
    if args.neo4j_database:
        neo4j.database = args.neo4j_database

    config = IndexConfig.create(
        dataset=args.dataset,
        index_name=args.index_name,
        index_root=args.index_root,
        neo4j=neo4j,
    )
    info = {
        "dataset": args.dataset,
        "index_name": args.index_name,
        "index_dir": str(config.index_dir),
        "chunks": count_jsonl(config.chunks_path),
        "triples": count_jsonl(config.triples_path),
        "vector_metadata": count_jsonl(config.vector_metadata_path),
        "faiss_vectors": 0,
        "faiss_dimension": None,
        "neo4j": {},
        "sample_triples": [],
    }
    if config.faiss_path.exists():
        index = faiss.read_index(str(config.faiss_path))
        info["faiss_vectors"] = int(index.ntotal)
        info["faiss_dimension"] = int(index.d)
    if config.manifest_path.exists():
        info["manifest"] = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    graph = Neo4jTripleStore(config.neo4j, config.dataset, config.index_name)
    try:
        info["neo4j"] = graph.stats()
        info["sample_triples"] = graph.export_triples(limit=args.sample_triples)
    finally:
        graph.close()
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
