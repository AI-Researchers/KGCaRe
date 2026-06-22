from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    BeautifulSoup = None  # type: ignore[assignment]

from kgcare.schemas import ConditionalQAQuestion, DatasetName, HotpotQAQuestion, SourceDocument


def stable_id(*parts: str) -> str:
    key = "::".join(parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _remove_hyperlinks(content: str) -> str:
    if BeautifulSoup is None:
        raise ImportError("Missing dependency 'beautifulsoup4'. Install with: pip install -r KGCaRe/requirements.txt")
    soup = BeautifulSoup(content, "html.parser")
    for anchor in soup.find_all("a"):
        anchor.unwrap()
    return str(soup)


def _conditionalqa_html_to_docs(html_text: str, filepath: Path) -> list[SourceDocument]:
    if BeautifulSoup is None:
        raise ImportError("Missing dependency 'beautifulsoup4'. Install with: pip install -r KGCaRe/requirements.txt")
    soup = BeautifulSoup(html_text, "html.parser")
    documents: list[SourceDocument] = []
    header_stack: list[str] = []
    current_text = ""

    def header_path() -> str:
        return "/".join(header_stack)

    def save_current_text() -> None:
        nonlocal current_text
        text = current_text.strip()
        if not text:
            return
        title = filepath.stem.replace("_", " ")
        header = header_path()
        doc_id = stable_id(str(filepath), header, text[:80])
        documents.append(
            SourceDocument(
                doc_id=doc_id,
                dataset="conditionalqa",
                title=title,
                text=text,
                source_path=str(filepath),
                metadata={
                    "File Name": str(filepath),
                    "Content Type": "text",
                    "Header Path": header,
                    "file_name": filepath.name,
                },
            )
        )
        current_text = ""

    for element in soup.descendants:
        if element.name in ["h1"]:
            save_current_text()
            header_level = int(element.name[1])
            header_text = element.get_text().strip()
            while len(header_stack) >= header_level:
                header_stack.pop()
            header_stack.append(header_text)
        elif element.name == "p":
            current_text += element.get_text() + "\n"
        elif element.name == "li":
            current_text += "- " + element.get_text() + "\n"
        elif element.name == "tr":
            cells = element.find_all("td")
            if cells:
                current_text += " | ".join(cell.get_text() for cell in cells) + "\n"

    save_current_text()
    if not documents and soup.get_text().strip():
        text = soup.get_text("\n").strip()
        documents.append(
            SourceDocument(
                doc_id=stable_id(str(filepath), text[:80]),
                dataset="conditionalqa",
                title=filepath.stem.replace("_", " "),
                text=text,
                source_path=str(filepath),
                metadata={"File Name": str(filepath), "file_name": filepath.name},
            )
        )
    return documents


def load_conditionalqa_documents(docs_path: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for path in sorted(docs_path.rglob("*.txt")):
        content = _remove_hyperlinks(path.read_text(encoding="utf-8"))
        docs.extend(_conditionalqa_html_to_docs(content, path))
    return docs


def load_hotpotqa_documents(docs_path: Path, dataset: DatasetName = "hotpotqa") -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for path in sorted(docs_path.rglob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = path.stem.replace("_", " ")
        docs.append(
            SourceDocument(
                doc_id=stable_id(str(path)),
                dataset=dataset,
                title=title,
                text=text,
                source_path=str(path),
                metadata={"file_path": str(path), "file_name": path.name},
            )
        )
    return docs


def load_documents(dataset: DatasetName, docs_path: Path) -> list[SourceDocument]:
    if dataset == "conditionalqa":
        return load_conditionalqa_documents(docs_path)
    if dataset == "hotpotqa":
        return load_hotpotqa_documents(docs_path, dataset="hotpotqa")
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_conditionalqa_questions(path: Path) -> list[ConditionalQAQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ConditionalQAQuestion.model_validate(item) for item in data]


def load_hotpotqa_questions(path: Path) -> list[HotpotQAQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [HotpotQAQuestion.model_validate(item) for item in data]


def iter_answerable_conditionalqa(path: Path) -> Iterable[ConditionalQAQuestion]:
    for question in load_conditionalqa_questions(path):
        if not question.not_answerable:
            yield question


def classify_conditionalqa_answer(answers: list[list]) -> str:
    if not answers:
        return "unanswerable"
    if any(ans[0] in ["yes", "no"] for ans in answers):
        return "yes/no_conditional" if any(ans[1] for ans in answers) else "yes/no"
    return "span_conditional" if any(ans[1] for ans in answers) else "span"
