from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from common import dataset_paths, iter_jsonl, write_jsonl
from output_schemas import ConditionalQAAnswer

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


def _normalize_conditional_answer(payload: dict[str, Any], question_type: str) -> dict[str, Any] | None:
    try:
        validated = ConditionalQAAnswer.model_validate(payload)
    except Exception:
        return None

    answer_type = validated.answer_type
    answers = [str(item).strip() for item in validated.answers if str(item).strip()]
    conditions = validated.conditions or []

    if answer_type == "yes_no":
        normalized_answers: list[str] = []
        for item in answers:
            token = item.lower().strip()
            if token.startswith("yes"):
                normalized_answers.append("yes")
            elif token.startswith("no"):
                normalized_answers.append("no")
        answers = normalized_answers[:1] or ["yes"]

    if answer_type == "span" and not answers:
        return None

    while len(conditions) < len(answers):
        conditions.append([])
    conditions = [list(map(str, condition_list)) for condition_list in conditions[: len(answers)]]

    if question_type.startswith("yes/no"):
        answer_type = "yes_no"
    else:
        answer_type = "span"

    return {
        "answer_type": answer_type,
        "answers": answers,
        "conditions": conditions,
        "rationale": validated.rationale,
    }


def _build_structured_prompt(question: str, question_type: str, strict: bool) -> str:
    strict_line = "Return ONLY valid JSON." if strict else "Prefer valid JSON and avoid extra text."
    if question_type.startswith("span"):
        task_rules = (
            "- For span type, answers must be exact contiguous spans copied from retrieved context.\n"
            "- Do NOT paraphrase, summarize, or use prior/world knowledge.\n"
            "- Do NOT output full sentences unless the span itself is a full sentence in context.\n"
            "- Prefer the shortest exact span that answers the question.\n"
            "- If exact span is not present in context, output \"unknown\" as the only answer.\n"
            "- Keep span answers concise (typically <= 10 words).\n"
        )
    else:
        task_rules = "- For yes/no type, answers must contain one lowercase item: yes or no.\n"

    return (
        f"{question}\n\n"
        "Respond with a single JSON object using this exact schema:\n"
        "{\n"
        '  "answer_type": "yes_no" | "span",\n'
        '  "answers": ["..."],\n'
        '  "conditions": [["..."]],\n'
        '  "rationale": "optional short note"\n'
        "}\n"
        "Rules:\n"
        f"- Question type is: {question_type}.\n"
        f"{task_rules}"
        "- conditions must align by index with answers; use [] when no condition.\n"
        f"- {strict_line}\n"
    )


def _parse_structured_output(raw_response: str, question_type: str) -> dict[str, Any] | None:
    candidate = _extract_json_candidate(raw_response)
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return _normalize_conditional_answer(payload, question_type)


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


async def ensure_ingested(rag: LightRAG, corpus_path: Path) -> None:
    for row in iter_jsonl(corpus_path):
        text = row["text"]
        await rag.ainsert(
            text,
            ids=row["id"],
            file_paths=row["metadata"].get("source_path"),
        )


async def run_queries(
    rag: LightRAG,
    queries_path: Path,
    predictions_path: Path,
    limit: int | None,
    structured_output: bool,
    structured_retries: int,
    max_total_tokens: int | None = None,
) -> None:
    rows = []
    for index, row in enumerate(iter_jsonl(queries_path), start=1):
        if limit is not None and index > limit:
            break

        query_text = row["composed_question"]
        raw_response = ""
        structured_answer = None
        parse_mode = "fallback"

        attempts = max(1, structured_retries) if structured_output else 1
        for attempt in range(attempts):
            prompt = query_text
            if structured_output:
                prompt = _build_structured_prompt(
                    question=query_text,
                    question_type=row["question_type"],
                    strict=attempt > 0,
                )

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
                structured_answer = _parse_structured_output(raw_response, row["question_type"])
                if structured_answer is not None:
                    parse_mode = "structured"
                    break

        rows.append(
            {
                "id": row["id"],
                "question": row["composed_question"],
                "question_type": row["question_type"],
                "reference_answers": row["reference_answers"],
                "raw_response": raw_response,
                "structured_answer": structured_answer,
                "parse_mode": parse_mode,
            }
        )
    write_jsonl(predictions_path, rows)


async def main_async(args: argparse.Namespace) -> None:
    paths = dataset_paths("ConditionalQA")
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
        embed_base_url = args.embedding_base_url or DEFAULT_EMBED_BASE_URL
        embed_api_key = args.embedding_api_key or os.getenv("EMBEDDING_BINDING_API_KEY", DEFAULT_API_KEY)
        embedding_dim = args.embedding_dim or int(os.getenv("EMBEDDING_DIM", "1024"))

    embedding_func = build_embedding_func(
        embed_model=embed_model_name,
        embed_base_url=embed_base_url,
        embed_api_key=embed_api_key,
        embedding_dim=embedding_dim,
        embedding_token_limit=args.embedding_token_limit,
    )

    if args.json_enforcement and args.ingest:
        raise ValueError("--json-enforcement cannot be used with --ingest (schema only applies to QA, not indexing)")

    llm_func_kwargs: dict = dict(
        model_name=llm_model_name,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    if args.json_enforcement:
        llm_func_kwargs["qa_response_format"] = ConditionalQAAnswer

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=partial(llm_model_func, **llm_func_kwargs),
        embedding_func=embedding_func,
        graph_storage=os.getenv("GRAPH_STORAGE", "Neo4JStorage"),
    )
    await rag.initialize_storages()
    try:
        if args.ingest:
            await ensure_ingested(rag, corpus_path)
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
    parser = argparse.ArgumentParser(description="Initial LightRAG ConditionalQA runner scaffold using vLLM-compatible endpoints.")
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
    parser.add_argument("--ingest", action="store_true", default=False)
    parser.add_argument("--max-total-tokens", type=int, default=None, dest="max_total_tokens",
                        help="Cap on total context tokens passed to the LLM. Use ~20000 for 32k-context models.")
    parser.add_argument("--structured-retries", type=int, default=2)
    parser.add_argument("--no-structured-output", action="store_false", dest="structured_output")
    parser.set_defaults(structured_output=True)
    parser.add_argument(
        "--json-enforcement",
        action="store_true",
        default=False,
        help="Pass ConditionalQAAnswer schema as response_format to the LLM for guided decoding (QA phase only; incompatible with --ingest).",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))
