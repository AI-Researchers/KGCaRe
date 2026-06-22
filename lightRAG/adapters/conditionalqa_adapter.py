from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import DATA_ROOT, dataset_paths, parse_doc_file, read_json, write_json, write_jsonl

DEFAULT_DOCS_DIR = DATA_ROOT / "docs_dev"
DEFAULT_REF_FILE = DATA_ROOT / "dev.json"


def classify_question(references: list[list[Any]]) -> str:
    if not references:
        return "unanswerable"
    if any(answer[0] in ["yes", "no"] for answer in references):
        return "yes/no_conditional" if any(answer[1] for answer in references) else "yes/no"
    return "span_conditional" if any(answer[1] for answer in references) else "span"


def build_corpus_rows(docs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(docs_dir.glob("*.txt")):
        parsed = parse_doc_file(path)
        rows.append(
            {
                "id": parsed["doc_id"],
                "title": parsed["title"],
                "text": parsed["content"],
                "metadata": {
                    "url": parsed["url"],
                    "source_path": parsed["source_path"],
                    "dataset": "ConditionalQA",
                },
            }
        )
    return rows



def build_query_rows(ref_file: Path) -> list[dict[str, Any]]:
    data = read_json(ref_file)
    rows: list[dict[str, Any]] = []
    for item in data:
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "scenario": item.get("scenario", ""),
                "composed_question": f"{item.get('scenario', '').strip()} {item['question']}".strip(),
                "question_type": classify_question(item["answers"]),
                "reference_answers": item["answers"],
                "not_answerable": item.get("not_answerable", False),
                "url": item.get("url", ""),
            }
        )
    return rows



def write_metadata(output_path: Path, docs_dir: Path, ref_file: Path, corpus_count: int, query_count: int) -> None:
    metadata = {
        "dataset": "ConditionalQA",
        "docs_dir": str(docs_dir),
        "ref_file": str(ref_file),
        "corpus_count": corpus_count,
        "query_count": query_count,
        "notes": [
            "Prepared for LightRAG local adapter use.",
            "Use existing core_utils evaluators for apples-to-apples comparison.",
        ],
    }
    write_json(output_path, metadata)



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ConditionalQA artifacts for LightRAG adapters.")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--ref-file", type=Path, default=DEFAULT_REF_FILE)
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
    write_metadata(metadata_out, args.docs_dir, args.ref_file, corpus_count, query_count)

    print(f"Prepared ConditionalQA corpus: {corpus_out} ({corpus_count} docs)")
    print(f"Prepared ConditionalQA queries: {queries_out} ({query_count} questions)")
    print(f"Metadata: {metadata_out}")


if __name__ == "__main__":
    main()
