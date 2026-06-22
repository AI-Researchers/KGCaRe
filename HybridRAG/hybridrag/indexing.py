from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import networkx as nx
from tqdm import tqdm

from hybridrag.data import ensure_dir, iter_jsonl, write_json, write_jsonl
from hybridrag.llm import invoke_structured
from hybridrag.prompts import build_refine_messages, build_triple_messages
from hybridrag.schemas import KGExtractionResult, KGTriple, RefinedChunk


VECTOR_STORE_DIR = "vector_faiss"
TRIPLE_STORE_DIR = "triple_faiss"
GRAPH_PATH = "graph.pkl"
TRIPLES_PATH = "triples.jsonl"
REFINED_PATH = "refined_chunks.jsonl"
MANIFEST_PATH = "index_manifest.json"


def make_embeddings(model: str, base_url: str | None = None, api_key: str | None = None):
    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, Any] = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)


def build_chunks(
    rows: list[dict[str, Any]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_prefix: str,
) -> list[Any]:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n<h1>", "\n<h2>", "\n<h3>", "\n<p>", "\n<li>", "\n", " ", ""],
    )
    documents: list[Document] = []
    for row in rows:
        chunks = splitter.split_text(row["text"])
        for index, chunk in enumerate(chunks):
            chunk_id = f"{chunk_prefix}-{row['id']}-{index:04d}"
            section_path = guess_section_path(row["title"], chunk)
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": row["id"],
                        "title": row["title"],
                        "url": row.get("url", ""),
                        "source_path": row.get("source_path", ""),
                        "chunk_id": chunk_id,
                        "chunk_index": index,
                        "section_path": section_path,
                    },
                )
            )
    return documents


def guess_section_path(title: str, text: str) -> str:
    headings = re.findall(r"<h[1-6]>\s*(.*?)\s*</h[1-6]>", text, flags=re.I | re.S)
    cleaned = [re.sub(r"\s+", " ", heading).strip() for heading in headings if heading.strip()]
    if cleaned:
        return " > ".join([title, *cleaned[-3:]])
    return title


def build_faiss_store(documents: list[Any], embeddings, persist_dir: Path) -> int:
    from langchain_community.vectorstores import FAISS

    if not documents:
        raise ValueError("Cannot build an empty FAISS store")
    ensure_dir(persist_dir)
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(str(persist_dir))
    return len(documents)


def load_corpus_rows(corpus_path: Path, limit_docs: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(corpus_path), start=1):
        if limit_docs is not None and index > limit_docs:
            break
        rows.append(row)
    return rows


def load_existing_refined(path: Path) -> dict[str, RefinedChunk]:
    if not path.exists():
        return {}
    refined: dict[str, RefinedChunk] = {}
    for row in iter_jsonl(path):
        refined[row["chunk_id"]] = RefinedChunk.model_validate(row["refined"])
    return refined


def load_existing_triples(path: Path) -> dict[str, list[KGTriple]]:
    if not path.exists():
        return {}
    triples_by_chunk: dict[str, list[KGTriple]] = {}
    for row in iter_jsonl(path):
        triple = KGTriple.model_validate(row["triple"])
        chunk_id = row.get("chunk_id") or triple.metadata.get("chunk_id", "")
        triples_by_chunk.setdefault(chunk_id, []).append(triple)
    return triples_by_chunk


def extract_refined_and_triples(
    *,
    llm,
    kg_chunks: list[Any],
    index_dir: Path,
    force_kg: bool,
    max_kg_chunks: int | None,
    structured_method: str = "function_calling",
    structured_retries: int = 2,
    skip_failed_chunks: bool = False,
) -> tuple[list[dict[str, Any]], list[KGTriple]]:
    refined_path = index_dir / REFINED_PATH
    triples_path = index_dir / TRIPLES_PATH

    if force_kg:
        refined_path.write_text("", encoding="utf-8")
        triples_path.write_text("", encoding="utf-8")

    refined_cache = load_existing_refined(refined_path)
    triples_cache = load_existing_triples(triples_path)
    refined_rows: list[dict[str, Any]] = []
    all_triples: list[KGTriple] = []
    failed_rows: list[dict[str, Any]] = []

    chunks = kg_chunks[:max_kg_chunks] if max_kg_chunks is not None else kg_chunks
    refined_file = refined_path.open("a", encoding="utf-8")
    triples_file = triples_path.open("a", encoding="utf-8")
    try:
        for document in tqdm(chunks, desc="Extracting KG", unit="chunk", dynamic_ncols=True):
            metadata = dict(document.metadata)
            chunk_id = metadata["chunk_id"]

            if chunk_id in refined_cache:
                refined = refined_cache[chunk_id]
            else:
                try:
                    refined, _, _ = invoke_structured_with_retries(
                        llm=llm,
                        schema=RefinedChunk,
                        messages=build_refine_messages(
                            title=metadata.get("title", ""),
                            url=metadata.get("url", ""),
                            chunk_id=chunk_id,
                            section_path=metadata.get("section_path", ""),
                            chunk_text=document.page_content,
                        ),
                        method=structured_method,
                        retries=structured_retries,
                    )
                except ValueError as exc:
                    if not skip_failed_chunks:
                        raise
                    failed_rows.append(
                        {
                            "chunk_id": chunk_id,
                            "metadata": metadata,
                            "stage": "refine",
                            "error": str(exc)[:1000],
                        }
                    )
                    continue
                refined_file.write(
                    json_line(
                        {
                            "chunk_id": chunk_id,
                            "metadata": metadata,
                            "refined": refined.model_dump(),
                        }
                    )
                )
                refined_file.flush()

            refined_rows.append({"chunk_id": chunk_id, "metadata": metadata, "refined": refined.model_dump()})

            if chunk_id in triples_cache:
                triples = triples_cache[chunk_id]
            else:
                try:
                    extraction, _, _ = invoke_structured_with_retries(
                        llm=llm,
                        schema=KGExtractionResult,
                        messages=build_triple_messages(
                            title=metadata.get("title", ""),
                            url=metadata.get("url", ""),
                            chunk_id=chunk_id,
                            section_path=metadata.get("section_path", ""),
                            refined_chunk_json=refined.model_dump_json(),
                        ),
                        method=structured_method,
                        retries=structured_retries,
                    )
                except ValueError as exc:
                    if not skip_failed_chunks:
                        raise
                    failed_rows.append(
                        {
                            "chunk_id": chunk_id,
                            "metadata": metadata,
                            "stage": "triples",
                            "error": str(exc)[:1000],
                        }
                    )
                    continue
                triples = []
                for index, triple in enumerate(extraction.triples):
                    triple.metadata.update(
                        {
                            "doc_id": str(metadata.get("doc_id", "")),
                            "title": str(metadata.get("title", "")),
                            "url": str(metadata.get("url", "")),
                            "chunk_id": chunk_id,
                            "section_path": str(metadata.get("section_path", "")),
                            "triple_id": f"triple-{chunk_id}-{index:03d}",
                        }
                    )
                    triples.append(triple)
                    triples_file.write(
                        json_line(
                            {
                                "chunk_id": chunk_id,
                                "triple_id": triple.metadata["triple_id"],
                                "triple": triple.model_dump(),
                            }
                        )
                    )
                triples_file.flush()
            all_triples.extend(triples)
    finally:
        refined_file.close()
        triples_file.close()

    if failed_rows:
        write_jsonl(index_dir / "failed_kg_chunks.jsonl", failed_rows)
    return refined_rows, all_triples


def invoke_structured_with_retries(
    *,
    llm,
    schema,
    messages: list[dict[str, str]],
    method: str,
    retries: int,
):
    last_error: Exception | None = None
    attempt_messages = messages
    for attempt in range(max(1, retries)):
        try:
            return invoke_structured(llm, schema, attempt_messages, method=method)
        except ValueError as exc:
            last_error = exc
            attempt_messages = build_retry_messages(messages, exc, attempt + 1)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Structured invocation failed without an exception")


def build_retry_messages(messages: list[dict[str, str]], error: Exception, attempt: int) -> list[dict[str, str]]:
    retry_messages = [dict(message) for message in messages]
    correction = (
        f"\n\nRetry {attempt}: the previous output failed validation: {str(error)[:600]}\n"
        "Return actual extracted data for the supplied chunk, not the JSON schema. "
        "Use exactly the requested top-level keys and return only one valid JSON object. "
        "If the previous output was too long or truncated, reduce the output aggressively: "
        "for triple extraction return at most 6 triples, use short evidence snippets, and omit lower-value facts."
    )
    for index in range(len(retry_messages) - 1, -1, -1):
        if retry_messages[index].get("role") == "user":
            retry_messages[index]["content"] = retry_messages[index].get("content", "") + correction
            return retry_messages
    retry_messages.append({"role": "user", "content": correction.strip()})
    return retry_messages


def json_line(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False) + "\n"


def normalize_node(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def build_graph(triples: list[KGTriple]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for triple in triples:
        head_key = normalize_node(triple.head)
        tail_key = normalize_node(triple.tail)
        graph.add_node(head_key, label=triple.head, type=triple.head_type)
        graph.add_node(tail_key, label=triple.tail, type=triple.tail_type)
        graph.add_edge(
            head_key,
            tail_key,
            key=triple.metadata.get("triple_id", ""),
            relation=triple.relation,
            evidence=triple.evidence,
            triple=triple.model_dump(),
            triple_id=triple.metadata.get("triple_id", ""),
        )
    return graph


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    with path.open("wb") as file_obj:
        pickle.dump(graph, file_obj)


def triple_documents(triples: list[KGTriple]) -> list[Any]:
    from langchain_core.documents import Document

    documents: list[Document] = []
    for triple in triples:
        triple_id = triple.metadata.get("triple_id", "")
        text = (
            f"{triple.head} -[{triple.relation}]-> {triple.tail}\n"
            f"Evidence: {triple.evidence}\n"
            f"Document: {triple.metadata.get('title', '')}\n"
            f"Section: {triple.metadata.get('section_path', '')}"
        )
        documents.append(
            Document(
                page_content=text,
                metadata={
                    **triple.metadata,
                    "triple_id": triple_id,
                    "head": triple.head,
                    "tail": triple.tail,
                    "relation": triple.relation,
                    "evidence": triple.evidence,
                    "head_key": normalize_node(triple.head),
                    "tail_key": normalize_node(triple.tail),
                },
            )
        )
    return documents


def build_index(
    *,
    corpus_path: Path,
    index_dir: Path,
    embeddings,
    llm,
    dataset_name: str = "ConditionalQA",
    vector_chunk_size: int = 1024,
    vector_chunk_overlap: int = 0,
    kg_chunk_size: int = 2024,
    kg_chunk_overlap: int = 204,
    force_vector: bool = False,
    force_kg: bool = False,
    limit_docs: int | None = None,
    max_kg_chunks: int | None = None,
    structured_method: str = "function_calling",
    structured_retries: int = 2,
    skip_failed_chunks: bool = False,
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_dir(index_dir)
    rows = load_corpus_rows(corpus_path, limit_docs=limit_docs)
    if not rows:
        raise ValueError(f"No corpus rows loaded from {corpus_path}")

    vector_chunks = build_chunks(
        rows,
        chunk_size=vector_chunk_size,
        chunk_overlap=vector_chunk_overlap,
        chunk_prefix="vec",
    )
    kg_chunks = build_chunks(
        rows,
        chunk_size=kg_chunk_size,
        chunk_overlap=kg_chunk_overlap,
        chunk_prefix="kg",
    )

    vector_store_dir = index_dir / VECTOR_STORE_DIR
    if force_vector or not (vector_store_dir / "index.faiss").exists():
        vector_count = build_faiss_store(vector_chunks, embeddings, vector_store_dir)
    else:
        vector_count = len(vector_chunks)

    _, triples = extract_refined_and_triples(
        llm=llm,
        kg_chunks=kg_chunks,
        index_dir=index_dir,
        force_kg=force_kg,
        max_kg_chunks=max_kg_chunks,
        structured_method=structured_method,
        structured_retries=structured_retries,
        skip_failed_chunks=skip_failed_chunks,
    )
    if not triples:
        raise ValueError("No KG triples were extracted; cannot build graph context index")

    graph = build_graph(triples)
    save_graph(graph, index_dir / GRAPH_PATH)
    triple_count = build_faiss_store(triple_documents(triples), embeddings, index_dir / TRIPLE_STORE_DIR)

    manifest = {
        "dataset": dataset_name,
        "corpus_path": str(corpus_path),
        "num_docs": len(rows),
        "num_vector_chunks": vector_count,
        "num_kg_chunks": len(kg_chunks[:max_kg_chunks] if max_kg_chunks is not None else kg_chunks),
        "num_triples": triple_count,
        "num_graph_nodes": graph.number_of_nodes(),
        "num_graph_edges": graph.number_of_edges(),
        "num_failed_kg_chunks": count_jsonl(index_dir / "failed_kg_chunks.jsonl"),
        "vector_chunk_size": vector_chunk_size,
        "vector_chunk_overlap": vector_chunk_overlap,
        "kg_chunk_size": kg_chunk_size,
        "kg_chunk_overlap": kg_chunk_overlap,
        **(manifest_extra or {}),
    }
    write_json(index_dir / MANIFEST_PATH, manifest)
    return manifest


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))
