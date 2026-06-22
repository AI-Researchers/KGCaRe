from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import DATA_ROOT, dataset_paths, read_json, write_json, write_jsonl

DEFAULT_DOCS_DIR = DATA_ROOT / "wiki_articles_supported_500"
DEFAULT_REF_FILE = DATA_ROOT / "stratified_hotpotqa_500sample_with_tag.json"


def build_corpus_rows(docs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "id": path.stem,
                "title": path.stem,
                "text": text,
                "source_path": str(path),
            }
        )
    return rows


def build_query_rows(ref_file: Path) -> list[dict[str, Any]]:
    data = read_json(ref_file)
    rows: list[dict[str, Any]] = []
    for item in data:
        rows.append(
            {
                "id": item["_id"],
                "question": item["question"],
                "reference_answer": item.get("answer", ""),
                "question_type": item.get("type", ""),
                "difficulty": item.get("level", ""),
                "supporting_facts": item.get("supporting_facts", []),
                "context": item.get("context", []),
            }
        )
    return rows


def write_metadata(path: Path, docs_dir: Path, ref_file: Path, corpus_count: int, query_count: int) -> None:
    payload = {
        "dataset": "HotpotQA",
        "docs_dir": str(docs_dir),
        "ref_file": str(ref_file),
        "corpus_count": corpus_count,
        "query_count": query_count,
    }
    write_json(path, payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA artifacts for HippoRAG adapters.")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--ref-file", type=Path, default=DEFAULT_REF_FILE)
    parser.add_argument("--corpus-out", type=Path, default=None)
    parser.add_argument("--queries-out", type=Path, default=None)
    parser.add_argument("--metadata-out", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = dataset_paths("HotpotQA")

    corpus_out = args.corpus_out or paths.corpus_path
    queries_out = args.queries_out or paths.queries_path
    metadata_out = args.metadata_out or paths.metadata_path

    corpus_rows = build_corpus_rows(args.docs_dir)
    query_rows = build_query_rows(args.ref_file)

    corpus_count = write_jsonl(corpus_out, corpus_rows)
    query_count = write_jsonl(queries_out, query_rows)
    write_metadata(metadata_out, args.docs_dir, args.ref_file, corpus_count, query_count)

    print(f"Prepared HotpotQA corpus: {corpus_out} ({corpus_count} docs)")
    print(f"Prepared HotpotQA queries: {queries_out} ({query_count} questions)")
    print(f"Metadata: {metadata_out}")


if __name__ == "__main__":
    main()
