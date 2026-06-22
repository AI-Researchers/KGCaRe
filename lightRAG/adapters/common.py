from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

LIGHTRAG_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LIGHTRAG_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "data"
WORKSPACE_DOCS_ROOT = WORKSPACE_ROOT / "workspace_docs"
DEFAULT_LIGHTRAG_DATA_ROOT = LIGHTRAG_ROOT / "adapter_data"
DEFAULT_LIGHTRAG_RUN_ROOT = LIGHTRAG_ROOT / "adapter_runs"


@dataclass(frozen=True)
class AdapterPaths:
    dataset_name: str
    corpus_path: Path
    queries_path: Path
    predictions_path: Path
    metadata_path: Path
    working_dir: Path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "item"


def dataset_paths(dataset_name: str) -> AdapterPaths:
    slug = slugify(dataset_name)
    dataset_root = ensure_dir(DEFAULT_LIGHTRAG_DATA_ROOT / slug)
    run_root = ensure_dir(DEFAULT_LIGHTRAG_RUN_ROOT / slug)
    working_dir = ensure_dir(LIGHTRAG_ROOT / "workspaces" / slug)
    return AdapterPaths(
        dataset_name=dataset_name,
        corpus_path=dataset_root / "corpus.jsonl",
        queries_path=dataset_root / "queries.jsonl",
        predictions_path=run_root / "predictions.jsonl",
        metadata_path=dataset_root / "metadata.json",
        working_dir=working_dir,
    )


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
