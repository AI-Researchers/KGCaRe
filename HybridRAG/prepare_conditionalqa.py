from __future__ import annotations

import argparse
from pathlib import Path

from hybridrag.data import DATA_ROOT, build_corpus_rows, build_query_rows, dataset_paths, write_json, write_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ConditionalQA JSONL files for HybridRAG.")
    parser.add_argument("--docs-dir", type=Path, default=DATA_ROOT / "docs_dev")
    parser.add_argument("--ref-file", type=Path, default=DATA_ROOT / "dev.json")
    parser.add_argument("--corpus-out", type=Path, default=None)
    parser.add_argument("--queries-out", type=Path, default=None)
    parser.add_argument("--metadata-out", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = dataset_paths("ConditionalQA")
    corpus_out = args.corpus_out or paths.corpus_path
    queries_out = args.queries_out or paths.queries_path
    metadata_out = args.metadata_out or paths.metadata_path

    corpus_rows = build_corpus_rows(args.docs_dir)
    query_rows = build_query_rows(args.ref_file)
    corpus_count = write_jsonl(corpus_out, corpus_rows)
    query_count = write_jsonl(queries_out, query_rows)
    write_json(
        metadata_out,
        {
            "dataset": "ConditionalQA",
            "docs_dir": str(args.docs_dir),
            "ref_file": str(args.ref_file),
            "corpus_count": corpus_count,
            "query_count": query_count,
        },
    )
    print(f"Prepared corpus: {corpus_out} ({corpus_count} docs)")
    print(f"Prepared queries: {queries_out} ({query_count} questions)")
    print(f"Metadata: {metadata_out}")


if __name__ == "__main__":
    main()
