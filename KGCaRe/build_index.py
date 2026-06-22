#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from kgcare.config import IndexConfig, Neo4jConfig
from kgcare.kg_builder import KGCaReIndexBuilder
from kgcare.llm import OpenAIChatClient, OpenAIEmbeddingClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a self-contained KGCaRe index.")
    parser.add_argument("--dataset", choices=["conditionalqa", "hotpotqa"], required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--kg-model", default="gpt-4o-2024-08-06")
    parser.add_argument("--kg-base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--kg-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--kg-output-mode", choices=["structured", "text"], default="structured")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--embedding-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--docs-path", type=Path)
    parser.add_argument("--questions-path", type=Path)
    parser.add_argument("--kg-docs-path", type=Path)
    parser.add_argument("--vector-docs-path", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--neo4j-uri")
    parser.add_argument("--neo4j-username")
    parser.add_argument("--neo4j-password")
    parser.add_argument("--neo4j-database")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--reset-graph", action="store_true")
    parser.add_argument("--overwrite-vectors", action="store_true")
    parser.add_argument("--overwrite-triples", action="store_true")
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
        kg_model=args.kg_model,
        embedding_model=args.embedding_model,
        index_root=args.index_root,
        docs_path=args.docs_path,
        questions_path=args.questions_path,
        kg_docs_path=args.kg_docs_path,
        vector_docs_path=args.vector_docs_path,
        neo4j=neo4j,
    )
    kg_api_key = args.kg_api_key or ("not_needed" if args.kg_base_url else None)
    embedding_api_key = args.embedding_api_key or None
    kg_llm = OpenAIChatClient(
        model=args.kg_model,
        api_key=kg_api_key,
        base_url=args.kg_base_url,
        max_tokens=args.max_tokens,
    )
    embedding_client = OpenAIEmbeddingClient(
        model=args.embedding_model,
        api_key=embedding_api_key,
        base_url=args.embedding_base_url,
    )
    builder = KGCaReIndexBuilder(config, kg_llm=kg_llm, embedding_client=embedding_client)
    manifest = builder.build(
        reset_graph=args.reset_graph,
        overwrite_vectors=args.overwrite_vectors,
        overwrite_triples=args.overwrite_triples,
        max_tokens=args.max_tokens,
        limit_docs=args.limit_docs,
        kg_output_mode=args.kg_output_mode,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
