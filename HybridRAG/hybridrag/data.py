from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
HYBRIDRAG_ROOT = WORKSPACE_ROOT / "HybridRAG"
DATA_ROOT = WORKSPACE_ROOT / "data"
ADAPTER_DATA_ROOT = HYBRIDRAG_ROOT / "adapter_data"
ADAPTER_RUN_ROOT = HYBRIDRAG_ROOT / "adapter_runs"
DEFAULT_INDEX_ROOT = HYBRIDRAG_ROOT / "indexes"


@dataclass(frozen=True)
class AdapterPaths:
    dataset_name: str
    corpus_path: Path
    queries_path: Path
    metadata_path: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "item"


def dataset_paths(dataset_name: str = "ConditionalQA") -> AdapterPaths:
    slug = slugify(dataset_name)
    dataset_root = ensure_dir(ADAPTER_DATA_ROOT / slug)
    return AdapterPaths(
        dataset_name=dataset_name,
        corpus_path=dataset_root / "corpus.jsonl",
        queries_path=dataset_root / "queries.jsonl",
        metadata_path=dataset_root / "metadata.json",
    )


def run_dir(dataset_name: str, provider: str, model_name: str, run_name: str) -> Path:
    return ensure_dir(
        ADAPTER_RUN_ROOT
        / slugify(dataset_name)
        / slugify(provider)
        / slugify(model_name)
        / slugify(run_name)
    )


def default_index_dir(index_name: str = "dev-gpt-4o-2024-08-06-shared", dataset_name: str = "ConditionalQA") -> Path:
    return ensure_dir(DEFAULT_INDEX_ROOT / slugify(dataset_name) / index_name)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_doc_file(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = lines[0].strip() if lines else path.stem
    url = lines[1].strip() if len(lines) > 1 else ""
    content = "\n".join(lines[2:]).strip() if len(lines) > 2 else text.strip()
    return {
        "doc_id": path.stem,
        "title": title,
        "url": url,
        "content": content,
        "source_path": str(path),
    }


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
        parsed = parse_doc_file(path)
        rows.append(
            {
                "id": parsed["doc_id"],
                "title": parsed["title"],
                "url": parsed["url"],
                "text": parsed["content"],
                "source_path": parsed["source_path"],
            }
        )
    return rows


def build_query_rows(ref_file: Path) -> list[dict[str, Any]]:
    data = read_json(ref_file)
    rows: list[dict[str, Any]] = []
    for item in data:
        scenario = item.get("scenario", "").strip()
        question = item.get("question", "").strip()
        composed_question = f"{scenario} {question}".strip()
        rows.append(
            {
                "id": item["id"],
                "url": item.get("url", ""),
                "scenario": scenario,
                "question": question,
                "composed_question": composed_question,
                "question_type": classify_question(item.get("answers", [])),
                "reference_answers": item.get("answers", []),
                "not_answerable": item.get("not_answerable", False),
            }
        )
    return rows


def build_hotpotqa_corpus_rows(docs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "id": path.stem,
                "title": path.stem,
                "url": "",
                "text": text,
                "source_path": str(path),
                "metadata": {
                    "source_path": str(path),
                    "dataset": "HotpotQA",
                },
            }
        )
    return rows


def classify_hotpotqa_question(item: dict[str, Any]) -> str:
    qtype = str(item.get("qtype") or item.get("type") or "").strip()
    if qtype:
        return qtype
    answer = str(item.get("answer", "")).strip().lower()
    if answer in {"yes", "no"}:
        return "yes/no"
    return "span"


def build_hotpotqa_query_rows(ref_file: Path) -> list[dict[str, Any]]:
    data = read_json(ref_file)
    rows: list[dict[str, Any]] = []
    for item in data:
        question = item["question"].strip()
        rows.append(
            {
                "id": item["_id"],
                "question": question,
                "composed_question": question,
                "question_type": classify_hotpotqa_question(item),
                "reference_answer": item.get("answer", ""),
                "entities": item.get("entities", []),
                "difficulty": item.get("level", ""),
                "supporting_facts": item.get("supporting_facts", []),
                "context": item.get("context", []),
            }
        )
    return rows
