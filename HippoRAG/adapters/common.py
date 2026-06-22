from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

HIPPORAG_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = HIPPORAG_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "data"
ADAPTER_DATA_ROOT = HIPPORAG_ROOT / "adapter_data"
ADAPTER_RUN_ROOT = HIPPORAG_ROOT / "adapter_runs"


@dataclass(frozen=True)
class AdapterPaths:
    dataset_name: str
    corpus_path: Path
    queries_path: Path
    metadata_path: Path


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
    dataset_root = ensure_dir(ADAPTER_DATA_ROOT / slug)
    return AdapterPaths(
        dataset_name=dataset_name,
        corpus_path=dataset_root / "corpus.jsonl",
        queries_path=dataset_root / "queries.jsonl",
        metadata_path=dataset_root / "metadata.json",
    )


def run_dir(dataset_name: str, provider: str, model_name: str, run_name: str | None = None) -> Path:
    slug_dataset = slugify(dataset_name)
    slug_provider = slugify(provider)
    slug_model = slugify(model_name)
    slug_run = slugify(run_name or "run")
    return ensure_dir(ADAPTER_RUN_ROOT / slug_dataset / slug_provider / slug_model / slug_run)


def parse_conditionalqa_doc(path: Path) -> dict[str, str]:
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
