from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import DATA_ROOT, dataset_paths, parse_conditionalqa_doc, read_json, write_json, write_jsonl

DEFAULT_DOCS_DIR = DATA_ROOT / "docs_dev"
DEFAULT_REF_FILE = DATA_ROOT / "dev.json"


def classify_question(reference_answers: list[list[Any]]) -> str:
    if not reference_answers:
        return "unanswerable"
    yes_no = any(answer[0] in ["yes", "no"] for answer in reference_answers)
    has_conditions = any(answer[1] for answer in reference_answers)
    if yes_no and has_conditions:
        return "yes/no_conditional"
    if yes_no:
        return "yes/no"
    if has_conditions:
        return "span_conditional"
    return "span"


def build_corpus_rows(docs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(docs_dir.glob("*.txt")):
        parsed = parse_conditionalqa_doc(path)
        rows.append(
            {
                "id": parsed["doc_id"],
                "title": parsed["title"],
                "text": parsed["content"],
                "source_path": parsed["source_path"],
                "url": parsed["url"],
            }
        )
    return rows


def build_query_rows(ref_file: Path) -> list[dict[str, Any]]:
    data = read_json(ref_file)
    rows: list[dict[str, Any]] = []
    for item in data:
        scenario = item.get("scenario", "").strip()
        question = item["question"].strip()
        composed_question = f"{scenario} {question}".strip()
        question_type = classify_question(item["answers"])
        rows.append(
            {
                "id": item["id"],
                "question": question,
                "scenario": scenario,
                "composed_question": composed_question,
                "question_type": question_type,
                "reference_answers": item["answers"],
                "not_answerable": item.get("not_answerable", False),
                "url": item.get("url", ""),
            }
        )
    return rows


def write_metadata(path: Path, docs_dir: Path, ref_file: Path, corpus_count: int, query_count: int) -> None:
    payload = {
        "dataset": "ConditionalQA",
        "docs_dir": str(docs_dir),
        "ref_file": str(ref_file),
        "corpus_count": corpus_count,
        "query_count": query_count,
    }
    write_json(path, payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ConditionalQA artifacts for HippoRAG adapters.")
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
