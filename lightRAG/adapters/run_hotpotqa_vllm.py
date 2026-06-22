from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from common import dataset_paths, iter_jsonl, write_jsonl
from output_schemas import HotpotQAAnswer

DEFAULT_MODEL = os.getenv("LLM_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
DEFAULT_BASE_URL = os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:8000/v1")
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
DEFAULT_EMBED_BASE_URL = os.getenv("EMBEDDING_BINDING_HOST", "http://127.0.0.1:8001/v1")
DEFAULT_API_KEY = os.getenv("LLM_BINDING_API_KEY", "not_needed")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
DEFAULT_OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def _extract_json_candidate(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_structured_output(raw_response: str) -> dict[str, Any] | None:
    candidate = _extract_json_candidate(raw_response)
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    try:
        validated = HotpotQAAnswer.model_validate(payload)
    except Exception:
        return None
    return {"answer": validated.answer.strip(), "rationale": validated.rationale}


def _build_structured_prompt(question: str, strict: bool) -> str:
    strict_line = "Return ONLY valid JSON." if strict else "Prefer valid JSON and avoid extra text."
    return (
        f"{question}\n\n"
        "Respond with a single JSON object using this exact schema:\n"
        "{\n"
        '  "answer": "final short answer",\n'
        '  "rationale": "optional short note"\n'
        "}\n"
        "Rules:\n"
        "- Keep answer concise and directly comparable to HotpotQA labels.\n"
        "- For yes/no questions output exactly 'yes' or 'no'.\n"
        f"- {strict_line}\n"
    )


async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    qa_response_format=None,
    **kwargs,
) -> str:
    # Inject QA schema for non-keyword-extraction calls.
    # openai_complete_if_cache overrides response_format with GPTKeywordExtractionFormat
    # when keyword_extraction=True, so keyword calls are unaffected.
    if qa_response_format is not None:
        kwargs.setdefault("response_format", qa_response_format)
    return await openai_complete_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )


def build_embedding_func(
    embed_model: str,
    embed_base_url: str,
    embed_api_key: str,
    embedding_dim: int,
    embedding_token_limit: int,
) -> EmbeddingFunc:
    return EmbeddingFunc(
        model_name=embed_model,
        embedding_dim=embedding_dim,
        max_token_size=embedding_token_limit,
        func=partial(
            openai_embed.func,
            model=embed_model,
            base_url=embed_base_url,
            api_key=embed_api_key,
        ),
    )


async def ensure_ingested(rag: LightRAG, corpus_path: Path, batch_size: int = 8) -> None:
    rows = list(iter_jsonl(corpus_path))
    print(f"Loaded {len(rows)} docs from corpus. Starting ingestion with batch_size={batch_size} ...", flush=True)
    bar = tqdm(total=len(rows), desc="Ingesting docs", unit="doc", dynamic_ncols=True)
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        ids = [r["id"] for r in batch]
        texts = [r["text"] for r in batch]
        paths = [r["metadata"].get("source_path") for r in batch]
        bar.set_postfix_str(f"batch {i // batch_size + 1}/{-(-len(rows) // batch_size)}: {ids[0][:30]}")
        await rag.ainsert(texts, ids=ids, file_paths=paths)
        bar.update(len(batch))
    bar.close()


async def run_queries(
    rag: LightRAG,
    queries_path: Path,
    predictions_path: Path,
    limit: int | None,
    structured_output: bool,
    structured_retries: int,
    max_total_tokens: int | None = None,
) -> None:
    all_rows = list(iter_jsonl(queries_path))
    if limit is not None:
        all_rows = all_rows[:limit]

    rows = []
    bar = tqdm(all_rows, desc="Running QA", unit="q", dynamic_ncols=True)
    for row in bar:
        bar.set_postfix_str(row["question"][:50])

        query_text = row["question"]
        raw_response = ""
        structured_answer = None
        parse_mode = "fallback"

        attempts = max(1, structured_retries) if structured_output else 1
        for attempt in range(attempts):
            prompt = query_text
            if structured_output:
                prompt = _build_structured_prompt(query_text, strict=attempt > 0)

            result = await rag.aquery(
                prompt,
                param=QueryParam(
                    mode="hybrid",
                    stream=False,
                    **({"max_total_tokens": max_total_tokens} if max_total_tokens is not None else {}),
                ),
            )
            raw_response = str(result)

            if structured_output:
                structured_answer = _parse_structured_output(raw_response)
                if structured_answer is not None:
                    parse_mode = "structured"
                    break

        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "raw_response": raw_response,
                "structured_answer": structured_answer,
                "parse_mode": parse_mode,
            }
        )
    write_jsonl(predictions_path, rows)


async def main_async(args: argparse.Namespace) -> None:
    paths = dataset_paths("HotpotQA")
    working_dir = args.working_dir or paths.working_dir
    corpus_path = args.corpus_path or paths.corpus_path
    queries_path = args.queries_path or paths.queries_path
    predictions_path = args.output_path or paths.predictions_path

    if args.provider == "openai":
        llm_model_name = args.llm_model or DEFAULT_OPENAI_MODEL
        llm_base_url = args.llm_base_url or DEFAULT_OPENAI_BASE_URL
        llm_api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY", "")
        embed_model_name = args.embedding_model or DEFAULT_OPENAI_EMBED_MODEL
        embed_base_url = args.embedding_base_url or DEFAULT_OPENAI_BASE_URL
        embed_api_key = args.embedding_api_key or os.getenv("OPENAI_API_KEY", "")
        embedding_dim = args.embedding_dim or 1536
    else:
        llm_model_name = args.llm_model or DEFAULT_MODEL
        llm_base_url = args.llm_base_url or DEFAULT_BASE_URL
        llm_api_key = args.llm_api_key or DEFAULT_API_KEY
        embed_model_name = args.embedding_model or DEFAULT_EMBED_MODEL
        # Fall back to OpenAI if no local embedding server is specified
        embed_base_url = args.embedding_base_url or os.getenv("EMBEDDING_BINDING_HOST") or DEFAULT_OPENAI_BASE_URL
        embed_api_key = args.embedding_api_key or os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY)
        embedding_dim = args.embedding_dim or int(os.getenv("EMBEDDING_DIM", "1536"))

    if args.json_enforcement and args.ingest:
        raise ValueError("--json-enforcement cannot be used with --ingest (schema only applies to QA, not indexing)")

    embedding_func = build_embedding_func(
        embed_model=embed_model_name,
        embed_base_url=embed_base_url,
        embed_api_key=embed_api_key,
        embedding_dim=embedding_dim,
        embedding_token_limit=args.embedding_token_limit,
    )

    llm_func_kwargs: dict = dict(
        model_name=llm_model_name,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    if args.json_enforcement:
        llm_func_kwargs["qa_response_format"] = HotpotQAAnswer

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=partial(llm_model_func, **llm_func_kwargs),
        embedding_func=embedding_func,
        graph_storage=args.graph_storage,
        max_parallel_insert=args.parallel_insert,
        enable_llm_cache=not args.no_llm_cache,
    )
    await rag.initialize_storages()
    try:
        if args.ingest:
            await ensure_ingested(rag, corpus_path, batch_size=args.parallel_insert)
        if not args.ingest_only:
            await run_queries(
                rag,
                queries_path,
                predictions_path,
                args.limit,
                args.structured_output,
                args.structured_retries,
                args.max_total_tokens,
            )
    finally:
        await rag.finalize_storages()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LightRAG HotpotQA runner with shared-index support.")
    parser.add_argument("--corpus-path", type=Path, default=None)
    parser.add_argument("--queries-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--working-dir", type=Path, default=None)
    parser.add_argument("--provider", choices=["openai-compatible", "openai"], default="openai-compatible")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--embedding-base-url", type=str, default=None)
    parser.add_argument("--embedding-api-key", type=str, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--embedding-token-limit", type=int, default=int(os.getenv("EMBEDDING_TOKEN_LIMIT", "4096")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parallel-insert", type=int, default=8, dest="parallel_insert",
                        help="Number of documents ingested in parallel (LightRAG max_parallel_insert). Default 8.")
    parser.add_argument("--ingest", action="store_true", default=False,
                        help="Run ingestion phase (builds graph from corpus).")
    parser.add_argument("--ingest-only", action="store_true", default=False,
                        help="Run ingestion only — skip QA. Use with --ingest.")
    parser.add_argument("--max-total-tokens", type=int, default=None, dest="max_total_tokens",
                        help="Cap on total context tokens passed to the LLM. Use ~20000 for 32k-context models.")
    parser.add_argument("--structured-retries", type=int, default=2)
    parser.add_argument("--no-structured-output", action="store_false", dest="structured_output")
    parser.set_defaults(structured_output=True)
    parser.add_argument(
        "--json-enforcement",
        action="store_true",
        default=False,
        help="Pass HotpotQAAnswer schema as response_format for guided decoding (QA phase only; incompatible with --ingest).",
    )
    parser.add_argument(
        "--graph-storage",
        type=str,
        default=os.getenv("GRAPH_STORAGE", "NetworkXStorage"),
        help="LightRAG graph storage backend. Default: NetworkXStorage (file-based). Use Neo4JStorage for Neo4j.",
    )
    parser.add_argument(
        "--no-llm-cache",
        action="store_true",
        default=False,
        help="Disable LLM response cache. REQUIRED for multi-model QA comparisons — the cache key is prompt-only and is shared across models.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))
