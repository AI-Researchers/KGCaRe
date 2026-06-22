#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from kgcare.config import IndexConfig, Neo4jConfig
from kgcare.graph_store import Neo4jTripleStore
from kgcare.llm import OpenAIChatClient, OpenAIEmbeddingClient
from kgcare.loaders import classify_conditionalqa_answer, load_conditionalqa_questions, load_hotpotqa_questions
from kgcare.qa import KGCaReQA, result_record
from kgcare.retriever import HybridKGCaReRetriever, KGCaReTraversal
from kgcare.schemas import ConditionalQAQuestion, HotpotQAQuestion
from kgcare.vector_store import FaissChunkStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QA over a KGCaRe index.")
    parser.add_argument("--dataset", choices=["conditionalqa", "hotpotqa"], required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--qa-model", default="gpt-4.1-nano")
    parser.add_argument("--qa-base-url")
    parser.add_argument("--qa-api-key")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-api-key")
    parser.add_argument("--mode", choices=["hybrid", "kg", "vector", "no_context"], default="hybrid")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--questions-path", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--neo4j-uri")
    parser.add_argument("--neo4j-username")
    parser.add_argument("--neo4j-password")
    parser.add_argument("--neo4j-database")
    parser.add_argument("--vector-top-k", type=int, default=10)
    parser.add_argument("--kg-max-depth", type=int, default=3)
    parser.add_argument("--kg-max-entities", type=int, default=20)
    parser.add_argument("--traversal-output-mode", choices=["structured", "text"], default="structured")
    parser.add_argument("--structured-method", choices=["parse", "json", "json_schema"], default="parse")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_neo4j_config(args: argparse.Namespace) -> Neo4jConfig:
    neo4j = Neo4jConfig()
    if args.neo4j_uri:
        neo4j.uri = args.neo4j_uri
    if args.neo4j_username:
        neo4j.username = args.neo4j_username
    if args.neo4j_password:
        neo4j.password = args.neo4j_password
    if args.neo4j_database:
        neo4j.database = args.neo4j_database
    return neo4j


def processed_ids(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("error"):
            continue
        if item.get("id"):
            seen.add(str(item["id"]))
        elif item.get("_id"):
            seen.add(str(item["_id"]))
    return seen


def iter_questions(config: IndexConfig) -> Iterable[tuple[str, str, Any, str, list[str]]]:
    if config.dataset == "conditionalqa":
        for item in load_conditionalqa_questions(config.dataset_paths.questions_path):
            if item.not_answerable:
                continue
            yield conditionalqa_payload(item)
        return
    for item in load_hotpotqa_questions(config.dataset_paths.questions_path):
        yield hotpotqa_payload(item)


def conditionalqa_payload(item: ConditionalQAQuestion) -> tuple[str, str, Any, str, list[str]]:
    return item.id, item.query_text, item.answers, classify_conditionalqa_answer(item.answers), []


def hotpotqa_payload(item: HotpotQAQuestion) -> tuple[str, str, Any, str, list[str]]:
    return item.id, item.question, item.answer, item.qtype, item.entities


def resolve_chat_connection(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.provider == "mistral":
        default_base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        default_api_key = os.getenv("MISTRAL_API_KEY")
    else:
        default_base_url = os.getenv("OPENAI_BASE_URL")
        default_api_key = os.getenv("OPENAI_API_KEY")
    return args.qa_base_url or default_base_url, args.qa_api_key or default_api_key


def resolve_embedding_connection(args: argparse.Namespace) -> tuple[str | None, str | None]:
    return (
        args.embedding_base_url or os.getenv("OPENAI_BASE_URL"),
        args.embedding_api_key or os.getenv("OPENAI_API_KEY"),
    )


def main() -> None:
    args = parse_args()
    config = IndexConfig.create(
        dataset=args.dataset,
        index_name=args.index_name,
        qa_model=args.qa_model,
        embedding_model=args.embedding_model,
        index_root=args.index_root,
        questions_path=args.questions_path,
        neo4j=build_neo4j_config(args),
    )

    run_dir = config.run_dir(args.qa_model, args.run_name, provider=args.provider)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "output.jsonl"
    if args.overwrite and output_path.exists():
        output_path.unlink()
    seen = processed_ids(output_path)

    qa_base_url, qa_api_key = resolve_chat_connection(args)
    llm = OpenAIChatClient(
        model=args.qa_model,
        api_key=qa_api_key,
        base_url=qa_base_url,
        max_tokens=args.max_tokens,
        structured_method=args.structured_method,
    )
    vector_store = None
    if args.mode in {"hybrid", "vector"}:
        embedding_base_url, embedding_api_key = resolve_embedding_connection(args)
        embedding_client = OpenAIEmbeddingClient(
            model=args.embedding_model,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
        )
        vector_store = FaissChunkStore(config.faiss_path, config.vector_metadata_path, embedding_client)
        vector_store.load()

    graph = None
    kg_retriever = None
    if args.mode in {"hybrid", "kg"}:
        graph = Neo4jTripleStore(config.neo4j, config.dataset, config.index_name)
        kg_retriever = KGCaReTraversal(
            graph_store=graph,
            llm=llm,
            max_depth=args.kg_max_depth,
            max_entities=args.kg_max_entities,
            traversal_output_mode=args.traversal_output_mode,
            include_trace=not args.no_trace,
        )

    retriever = HybridKGCaReRetriever(
        vector_store=vector_store,
        kg_retriever=kg_retriever,
        vector_top_k=args.vector_top_k,
    )
    qa = KGCaReQA(llm=llm, max_tokens=args.max_tokens)
    questions = list(iter_questions(config))
    if args.limit is not None:
        questions = questions[: args.limit]

    run_config = {
        "dataset": args.dataset,
        "index_name": args.index_name,
        "mode": args.mode,
        "provider": args.provider,
        "qa_model": args.qa_model,
        "qa_base_url": qa_base_url,
        "embedding_model": args.embedding_model,
        "traversal_output_mode": args.traversal_output_mode,
        "structured_method": args.structured_method,
        "include_trace": not args.no_trace,
        "questions_path": str(config.dataset_paths.questions_path),
        "index_dir": str(config.index_dir),
        "output_path": str(output_path),
        "total_questions": len(questions),
        "already_processed": len(seen),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    try:
        with output_path.open("a", encoding="utf-8") as output_file:
            for question_id, question, gold_answer, qtype, topic_entities in tqdm(
                questions,
                desc=f"KGCaRe QA {args.dataset} {args.qa_model}",
                unit="q",
            ):
                if question_id in seen:
                    continue
                try:
                    retrieval = retriever.retrieve(
                        question_id=question_id,
                        question=question,
                        mode=args.mode,
                        topic_entities=topic_entities,
                    )
                    structured, raw_response = qa.answer(
                        dataset=args.dataset,
                        question=question,
                        retrieval=retrieval,
                        qtype=qtype,
                    )
                    record = result_record(
                        dataset=args.dataset,
                        question_id=question_id,
                        question=question,
                        gold_answer=gold_answer,
                        qtype=qtype,
                        retrieval=retrieval,
                        structured=structured,
                        raw_response=raw_response,
                    )
                except Exception as exc:
                    record = {
                        "id": question_id,
                        "_id": question_id,
                        "question": question,
                        "question_type": qtype,
                        "answer": [] if args.dataset == "conditionalqa" else "",
                        "predicted_answer": [] if args.dataset == "conditionalqa" else "",
                        "gold_answer": gold_answer,
                        "error": str(exc),
                    }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
    finally:
        if graph is not None:
            graph.close()

    print(json.dumps({"output": str(output_path), "processed": len(processed_ids(output_path))}, indent=2))


if __name__ == "__main__":
    main()
