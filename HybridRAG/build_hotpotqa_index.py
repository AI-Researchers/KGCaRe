from __future__ import annotations

import argparse
import os
from pathlib import Path

from hybridrag.data import (
    DATA_ROOT,
    build_hotpotqa_corpus_rows,
    dataset_paths,
    default_index_dir,
    write_json,
    write_jsonl,
)
from hybridrag.indexing import build_index, make_embeddings
from hybridrag.llm import make_chat_model


DEFAULT_DOCS_DIR = DATA_ROOT / "wiki_articles_supported_500"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the shared HybridRAG HotPotQA index.")
    parser.add_argument("--corpus-path", type=Path, default=None)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--kg-provider", choices=["openai", "openai-compatible", "vllm"], default="vllm")
    parser.add_argument("--kg-model", type=str, default=os.getenv("KG_MODEL", "Qwen3.6-27B"))
    parser.add_argument("--kg-base-url", type=str, default=None)
    parser.add_argument("--kg-api-key", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embedding-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--embedding-api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--vector-chunk-size", type=int, default=1024)
    parser.add_argument("--vector-chunk-overlap", type=int, default=0)
    parser.add_argument("--kg-chunk-size", type=int, default=2024)
    parser.add_argument("--kg-chunk-overlap", type=int, default=204)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--max-kg-chunks", type=int, default=None)
    parser.add_argument("--force-vector", action="store_true", default=False)
    parser.add_argument("--force-kg", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--structured-method",
        choices=["auto", "function_calling", "json"],
        default="auto",
        help="KG extraction structured output method. auto uses JSON-only prompting for vLLM.",
    )
    parser.add_argument("--structured-retries", type=int, default=2)
    parser.add_argument(
        "--skip-failed-kg-chunks",
        action="store_true",
        default=False,
        help="After retries, record malformed KG chunks and continue building the index.",
    )
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


def ensure_corpus(corpus_path: Path, docs_dir: Path) -> None:
    if corpus_path.exists():
        return
    rows = build_hotpotqa_corpus_rows(docs_dir)
    count = write_jsonl(corpus_path, rows)
    write_json(
        dataset_paths("HotPotQA").metadata_path,
        {
            "dataset": "HotPotQA",
            "docs_dir": str(docs_dir),
            "corpus_path": str(corpus_path),
            "corpus_count": count,
        },
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = dataset_paths("HotPotQA")
    corpus_path = args.corpus_path or paths.corpus_path
    index_dir = args.index_dir or default_index_dir("stratified-qwen-shared", dataset_name="HotPotQA")
    provider = "vllm" if args.kg_provider == "vllm" else args.kg_provider
    kg_base_url = args.kg_base_url or default_base_url(provider)
    kg_api_key = args.kg_api_key or default_api_key(provider)
    structured_method = resolve_structured_method(provider, args.structured_method)
    ensure_corpus(corpus_path, args.docs_dir)

    embeddings = make_embeddings(
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
    )
    llm = make_chat_model(
        model=args.kg_model,
        base_url=kg_base_url,
        api_key=kg_api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    manifest = build_index(
        corpus_path=corpus_path,
        index_dir=index_dir,
        embeddings=embeddings,
        llm=llm,
        dataset_name="HotPotQA",
        vector_chunk_size=args.vector_chunk_size,
        vector_chunk_overlap=args.vector_chunk_overlap,
        kg_chunk_size=args.kg_chunk_size,
        kg_chunk_overlap=args.kg_chunk_overlap,
        force_vector=args.force_vector,
        force_kg=args.force_kg,
        limit_docs=args.limit_docs,
        max_kg_chunks=args.max_kg_chunks,
        structured_method=structured_method,
        structured_retries=args.structured_retries,
        skip_failed_chunks=args.skip_failed_kg_chunks,
        manifest_extra={
            "kg_provider": provider,
            "kg_model": args.kg_model,
            "kg_base_url": kg_base_url,
            "embedding_model": args.embedding_model,
            "structured_method": structured_method,
            "structured_retries": args.structured_retries,
            "skip_failed_kg_chunks": args.skip_failed_kg_chunks,
        },
    )
    print(f"Index directory: {index_dir}")
    print(f"Manifest: {index_dir / 'index_manifest.json'}")
    print(manifest)


if __name__ == "__main__":
    main()
