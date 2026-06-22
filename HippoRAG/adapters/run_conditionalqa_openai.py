from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import dataset_paths
from conditionalqa_adapter import build_corpus_rows, build_query_rows
from run_utils import load_corpus_docs, load_query_rows, run_hipporag


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HippoRAG on ConditionalQA with OpenAI models.")
    parser.add_argument("--llm-model", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--llm-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--embedding-model", type=str, default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embedding-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--run-name", type=str, default="smoke")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--ref-file", type=Path, default=None)
    parser.add_argument("--prepare-data", action="store_true", default=False)
    parser.add_argument("--ingest", action="store_true", default=False)
    parser.add_argument("--force-index-from-scratch", action="store_true", default=False)
    parser.add_argument("--force-openie-from-scratch", action="store_true", default=False)
    parser.add_argument("--retrieval-top-k", type=int, default=30)
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def format_query(row: dict) -> str:
    question_type = row.get("question_type", "span")
    question = row.get("composed_question") or row.get("question") or ""
    return f"[Question Type: {question_type}] {question}".strip()


def maybe_prepare(args: argparse.Namespace) -> None:
    if not args.prepare_data:
        return
    paths = dataset_paths("ConditionalQA")
    docs_dir = args.docs_dir or (Path(__file__).resolve().parents[2] / "data" / "docs_dev")
    ref_file = args.ref_file or (Path(__file__).resolve().parents[2] / "data" / "dev.json")
    corpus_rows = build_corpus_rows(docs_dir)
    query_rows = build_query_rows(ref_file)

    from common import write_jsonl, write_json

    corpus_count = write_jsonl(paths.corpus_path, corpus_rows)
    query_count = write_jsonl(paths.queries_path, query_rows)
    write_json(
        paths.metadata_path,
        {
            "dataset": "ConditionalQA",
            "docs_dir": str(docs_dir),
            "ref_file": str(ref_file),
            "corpus_count": corpus_count,
            "query_count": query_count,
        },
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    maybe_prepare(args)

    paths = dataset_paths("ConditionalQA")
    docs = load_corpus_docs(paths.corpus_path)
    query_rows = load_query_rows(paths.queries_path, args.limit)

    save_path, predictions_path, config_path = run_hipporag(
        dataset_key="conditionalqa",
        provider="openai",
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        query_rows=query_rows,
        docs=docs,
        query_builder=format_query,
        run_name=args.run_name,
        force_index_from_scratch=args.force_index_from_scratch,
        force_openie_from_scratch=args.force_openie_from_scratch,
        retrieval_top_k=args.retrieval_top_k,
        qa_top_k=args.qa_top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        ingest=args.ingest,
    )

    print(f"Run directory: {save_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
