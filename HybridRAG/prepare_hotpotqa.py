from __future__ import annotations

import argparse
from pathlib import Path

from hybridrag.data import (
    DATA_ROOT,
    build_hotpotqa_corpus_rows,
    build_hotpotqa_query_rows,
    dataset_paths,
    write_json,
    write_jsonl,
)


DEFAULT_DOCS_DIR = DATA_ROOT / "wiki_articles_supported_500"
DEFAULT_REF_FILE = DATA_ROOT / "stratified_hotpotqa_500sample_with_tag.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare stratified HotPotQA artifacts for HybridRAG.")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--ref-file", type=Path, default=DEFAULT_REF_FILE)
    parser.add_argument("--corpus-out", type=Path, default=None)
    parser.add_argument("--queries-out", type=Path, default=None)
    parser.add_argument("--metadata-out", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = dataset_paths("HotPotQA")

    corpus_out = args.corpus_out or paths.corpus_path
    queries_out = args.queries_out or paths.queries_path
    metadata_out = args.metadata_out or paths.metadata_path

    corpus_rows = build_hotpotqa_corpus_rows(args.docs_dir)
    query_rows = build_hotpotqa_query_rows(args.ref_file)

    corpus_count = write_jsonl(corpus_out, corpus_rows)
    query_count = write_jsonl(queries_out, query_rows)
    question_type_counts: dict[str, int] = {}
    for row in query_rows:
        question_type_counts[row["question_type"]] = question_type_counts.get(row["question_type"], 0) + 1

    write_json(
        metadata_out,
        {
            "dataset": "HotPotQA",
            "docs_dir": str(args.docs_dir),
            "ref_file": str(args.ref_file),
            "corpus_path": str(corpus_out),
            "queries_path": str(queries_out),
            "corpus_count": corpus_count,
            "query_count": query_count,
            "question_type_counts": question_type_counts,
            "notes": [
                "Prepared from the stratified 500-question HotPotQA sample.",
                "Corpus uses the supported 500-sample wiki article directory.",
            ],
        },
    )

    print(f"Prepared HotPotQA corpus: {corpus_out} ({corpus_count} docs)")
    print(f"Prepared HotPotQA queries: {queries_out} ({query_count} questions)")
    print(f"Question types: {question_type_counts}")
    print(f"Metadata: {metadata_out}")


if __name__ == "__main__":
    main()
