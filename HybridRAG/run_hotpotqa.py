from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from hybridrag.data import dataset_paths, default_index_dir, iter_jsonl, run_dir, write_json
from hybridrag.indexing import make_embeddings
from hybridrag.llm import invoke_structured, make_chat_model
from hybridrag.prompts import build_hotpotqa_messages
from hybridrag.retrieval import HybridRetriever
from hybridrag.schemas import HotpotQAAnswer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HybridRAG QA on the stratified HotPotQA sample.")
    parser.add_argument("--queries-path", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--provider", choices=["openai", "openai-compatible", "vllm"], default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embedding-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--embedding-api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--run-name", type=str, default="full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vector-fetch-k", type=int, default=20)
    parser.add_argument("--vector-context-k", type=int, default=4)
    parser.add_argument("--graph-seed-k", type=int, default=8)
    parser.add_argument("--graph-depth", type=int, default=1)
    parser.add_argument("--graph-context-k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--structured-method",
        choices=["auto", "function_calling", "json"],
        default="auto",
        help="Structured output method. auto uses function_calling for OpenAI and JSON-only prompting for vLLM.",
    )
    parser.add_argument("--structured-retries", type=int, default=2)
    parser.add_argument(
        "--on-parse-error",
        choices=["auto", "raise", "empty"],
        default="auto",
        help="How to handle repeated structured-output parse failures. auto raises for OpenAI and records an empty answer for vLLM.",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser


def default_base_url(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")


def default_api_key(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    return os.getenv("VLLM_API_KEY", "not_needed")


def resolve_structured_method(provider: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "json" if provider == "vllm" else "function_calling"


def resolve_parse_error_mode(provider: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "empty" if provider == "vllm" else "raise"


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed: set[str] = set()
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            processed.add(json.loads(line)["id"])
    return processed


def load_query_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(path), start=1):
        if limit is not None and index > limit:
            break
        rows.append(row)
    return rows


def normalize_hotpot_answer(answer: str, gold_answer: str) -> str:
    cleaned = answer.strip()
    if gold_answer.strip().lower() in {"yes", "no"}:
        lowered = cleaned.lower()
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"
    return cleaned


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = dataset_paths("HotPotQA")
    queries_path = args.queries_path or paths.queries_path
    index_dir = args.index_dir or default_index_dir("stratified-qwen-shared", dataset_name="HotPotQA")
    provider = "vllm" if args.provider == "vllm" else args.provider
    qa_run_dir = run_dir("HotPotQA", provider, args.llm_model, args.run_name)
    output_path = args.output_path or (qa_run_dir / "predictions.jsonl")

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}. Run HybridRAG/prepare_hotpotqa.py first.")

    llm_base_url = args.llm_base_url or default_base_url(provider)
    llm_api_key = args.llm_api_key or default_api_key(provider)
    embeddings = make_embeddings(
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
    )
    retriever = HybridRetriever(index_dir=index_dir, embeddings=embeddings)
    llm = make_chat_model(
        model=args.llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    structured_method = resolve_structured_method(provider, args.structured_method)
    parse_error_mode = resolve_parse_error_mode(provider, args.on_parse_error)
    structured_retries = max(1, args.structured_retries)

    write_json(
        qa_run_dir / "run_config.json",
        {
            "dataset": "HotPotQA",
            "provider": provider,
            "llm_model": args.llm_model,
            "llm_base_url": llm_base_url,
            "embedding_model": args.embedding_model,
            "embedding_base_url": args.embedding_base_url,
            "index_dir": str(index_dir),
            "queries_path": str(queries_path),
            "output_path": str(output_path),
            "run_name": args.run_name,
            "vector_fetch_k": args.vector_fetch_k,
            "vector_context_k": args.vector_context_k,
            "graph_seed_k": args.graph_seed_k,
            "graph_depth": args.graph_depth,
            "graph_context_k": args.graph_context_k,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "structured_method": structured_method,
            "structured_retries": structured_retries,
            "on_parse_error": parse_error_mode,
        },
    )

    rows = load_query_rows(queries_path, args.limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = set() if args.overwrite else load_processed_ids(output_path)
    mode = "w" if args.overwrite else "a"

    with output_path.open(mode, encoding="utf-8") as file_obj:
        for row in tqdm(rows, desc=f"HybridRAG HotPotQA {args.llm_model}", unit="q", dynamic_ncols=True):
            if row["id"] in processed:
                continue
            retrieval_query = row.get("composed_question") or row.get("question", "")
            context = retriever.retrieve(
                retrieval_query,
                vector_fetch_k=args.vector_fetch_k,
                vector_context_k=args.vector_context_k,
                graph_seed_k=args.graph_seed_k,
                graph_depth=args.graph_depth,
                graph_context_k=args.graph_context_k,
            )
            messages = build_hotpotqa_messages(
                question_type=row.get("question_type", "span"),
                question=row.get("question", ""),
                vector_context=context.vector_context,
                graph_context=context.graph_context,
            )

            structured: HotpotQAAnswer | None = None
            raw_response = ""
            parse_mode = ""
            parse_error = ""
            for attempt in range(1, structured_retries + 1):
                try:
                    structured, raw_response, parse_mode = invoke_structured(
                        llm,
                        HotpotQAAnswer,
                        messages,
                        method=structured_method,
                    )
                    break
                except Exception as exc:
                    if not isinstance(exc, ValueError):
                        raise
                    parse_error = f"{type(exc).__name__}: {exc}"
                    if attempt >= structured_retries and parse_error_mode == "raise":
                        raise

            if structured is None:
                structured = HotpotQAAnswer(answer="", rationale=parse_error[:1000])
                raw_response = parse_error
                parse_mode = "parse_error_empty"

            reference_answer = row.get("reference_answer", "")
            normalized_answer = normalize_hotpot_answer(structured.answer, reference_answer)
            output_row = {
                "id": row["id"],
                "question": row.get("question", ""),
                "composed_question": retrieval_query,
                "question_type": row.get("question_type", ""),
                "reference_answer": reference_answer,
                "entities": row.get("entities", []),
                "structured_answer": structured.model_dump(),
                "normalized_answer": normalized_answer,
                "raw_response": raw_response,
                "parse_mode": parse_mode,
                "vector_context": context.vector_items,
                "graph_context": context.graph_items,
            }
            file_obj.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            file_obj.flush()

    print(f"Predictions: {output_path}")
    print(f"Run directory: {qa_run_dir}")


if __name__ == "__main__":
    main()
