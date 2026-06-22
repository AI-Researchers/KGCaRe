from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Literal

from tqdm import tqdm

from kgcare.config import IndexConfig
from kgcare.graph_store import Neo4jTripleStore
from kgcare.llm import OpenAIChatClient, OpenAIEmbeddingClient
from kgcare.loaders import load_documents, load_hotpotqa_documents
from kgcare.prompts import kg_prompt_chain
from kgcare.schemas import (
    ChunkRecord,
    IndexManifest,
    KGStructuredTriples,
    SourceDocument,
    TripleRecord,
    read_jsonl_models,
    write_jsonl_models,
)
from kgcare.vector_store import FaissChunkStore

KGOutputMode = Literal["text", "structured"]


def make_chunk_id(dataset: str, doc_id: str, ordinal: int) -> str:
    return f"{dataset}-{doc_id}-{ordinal:04d}"


def documents_to_chunks(documents: list[SourceDocument]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for ordinal, doc in enumerate(documents):
        chunks.append(
            ChunkRecord(
                chunk_id=make_chunk_id(doc.dataset, doc.doc_id, ordinal),
                doc_id=doc.doc_id,
                dataset=doc.dataset,
                title=doc.title,
                text=doc.text,
                source_path=doc.source_path,
                metadata=doc.metadata,
                start_char=0,
                end_char=len(doc.text),
            )
        )
    return chunks


def parse_triplet_response(response: str, max_length: int = 1000) -> list[tuple[str, str, str]]:
    """Parse final KG prompt output using the legacy KGCaRe/LlamaIndex logic."""
    knowledge_strs = response.strip().split("\n")
    results: list[tuple[str, str, str]] = []
    for text in knowledge_strs:
        if "(" not in text or ")" not in text or text.index(")") < text.index("("):
            continue
        triplet_part = text[text.index("(") + 1 : text.index(")")]
        tokens = triplet_part.split(",")
        if len(tokens) != 3:
            continue
        if any(len(s.encode("utf-8")) > max_length for s in tokens):
            continue
        subj, pred, obj = map(str.strip, tokens)
        if not subj or not pred or not obj:
            continue
        subj, pred, obj = (entity.strip('"').capitalize() for entity in [subj, pred, obj])
        pred = pred.strip("'").strip()
        invalid_chars = [
            "&",
            "*",
            ":",
            "WHERE",
            "]",
            "{",
            "|",
            "}",
            "[",
            "(",
            ")",
            "<",
            ">",
            "#",
            "@",
            "!",
            "$",
            "%",
            "^",
            "+",
            "=",
            "?",
            "/",
            "\\",
            "`",
            "~",
            ";",
            ",",
            ".",
            "'",
            '"',
            "\n",
            "\t",
            "\r",
        ]
        for char in invalid_chars:
            pred = pred.replace(char, "")
        pred = " ".join(pred.split())
        results.append((subj, pred, obj))
    return results


def sanitize_relation(predicate: str) -> str:
    invalid_chars = [
        "&",
        "*",
        ":",
        "WHERE",
        "]",
        "{",
        "|",
        "}",
        "[",
        "(",
        ")",
        "<",
        ">",
        "#",
        "@",
        "!",
        "$",
        "%",
        "^",
        "+",
        "=",
        "?",
        "/",
        "\\",
        "`",
        "~",
        ";",
        ",",
        ".",
        "'",
        '"',
        "\n",
        "\t",
        "\r",
    ]
    cleaned = predicate.strip()
    for char in invalid_chars:
        cleaned = cleaned.replace(char, "")
    return " ".join(cleaned.split())


def normalize_structured_triples(payload: KGStructuredTriples, max_length: int = 1000) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in payload.triples:
        subject = item.subject.strip()
        predicate = sanitize_relation(item.predicate)
        obj = item.object.strip()
        if not subject or not predicate or not obj:
            continue
        if any(len(part.encode("utf-8")) > max_length for part in [subject, predicate, obj]):
            continue
        triple = (subject, predicate, obj)
        if triple in seen:
            continue
        seen.add(triple)
        triples.append(triple)
    return triples


def find_evidence_sentence(text: str, triple: tuple[str, str, str]) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]
    if not sentences:
        return ""
    query_terms = {
        term
        for part in triple
        for term in re.findall(r"[A-Za-z0-9£$%-]+", part.lower())
        if len(term) > 2
    }
    if not query_terms:
        return ""
    best_sentence = ""
    best_score = 0
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for term in query_terms if term in sentence_lower)
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence if best_score > 0 else ""


def triple_id(dataset: str, index_name: str, chunk_id: str, triple: tuple[str, str, str]) -> str:
    raw = "::".join([dataset, index_name, chunk_id, *triple])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def run_three_step_kg_extraction(
    llm: OpenAIChatClient,
    dataset: str,
    text: str,
    max_tokens: int = 2048,
    output_mode: KGOutputMode = "structured",
) -> tuple[str, str, str, list[tuple[str, str, str]]]:
    prompt1, prompt2, prompt3 = kg_prompt_chain(dataset)  # type: ignore[arg-type]
    step1 = llm.chat_text(prompt1.format(text=text), max_tokens=max_tokens)
    step2 = llm.chat_text(prompt2.format(text=text, intermediate_extracted=step1), max_tokens=max_tokens)
    step3_prompt = prompt3.format(text=text, intermediate_extracted=step2)
    if output_mode == "structured":
        try:
            structured, raw = llm.structured(
                KGStructuredTriples,
                [
                    {
                        "role": "system",
                        "content": (
                            "Return the final knowledge graph triples using the provided structured schema. "
                            "Each item must preserve the subject, predicate, and object from the final refinement task."
                        ),
                    },
                    {"role": "user", "content": step3_prompt},
                ],
                max_tokens=max_tokens,
            )
            return step1, step2, raw, normalize_structured_triples(structured)
        except Exception:
            step3 = llm.chat_text(step3_prompt, max_tokens=max_tokens)
            return step1, step2, step3, parse_triplet_response(step3)
    step3 = llm.chat_text(step3_prompt, max_tokens=max_tokens)
    return step1, step2, step3, parse_triplet_response(step3)


def _load_existing_triples(path: Path) -> list[TripleRecord]:
    return [record for record in read_jsonl_models(path, TripleRecord)]  # type: ignore[list-item]


class KGCaReIndexBuilder:
    def __init__(
        self,
        config: IndexConfig,
        kg_llm: OpenAIChatClient,
        embedding_client: OpenAIEmbeddingClient,
    ) -> None:
        self.config = config
        self.kg_llm = kg_llm
        self.embedding_client = embedding_client

    def load_index_documents(self) -> tuple[list[SourceDocument], list[SourceDocument]]:
        if self.config.dataset == "hotpotqa":
            kg_docs_path = self.config.dataset_paths.kg_docs_path or self.config.dataset_paths.docs_path
            vector_docs_path = self.config.dataset_paths.vector_docs_path or self.config.dataset_paths.docs_path
            kg_docs = load_hotpotqa_documents(kg_docs_path, dataset="hotpotqa")
            vector_docs = load_hotpotqa_documents(vector_docs_path, dataset="hotpotqa")
            return kg_docs, vector_docs
        docs = load_documents(self.config.dataset, self.config.dataset_paths.docs_path)
        return docs, docs

    def build(
        self,
        reset_graph: bool = False,
        overwrite_vectors: bool = False,
        overwrite_triples: bool = False,
        max_tokens: int = 2048,
        limit_docs: int | None = None,
        kg_output_mode: KGOutputMode = "structured",
    ) -> IndexManifest:
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        kg_docs, vector_docs = self.load_index_documents()
        if limit_docs is not None:
            kg_docs = kg_docs[:limit_docs]
            vector_docs = vector_docs[:limit_docs]
        kg_chunks = documents_to_chunks(kg_docs)
        vector_chunks = documents_to_chunks(vector_docs)
        all_chunks_by_id = {chunk.chunk_id: chunk for chunk in vector_chunks}
        all_chunks_by_id.update({chunk.chunk_id: chunk for chunk in kg_chunks})
        all_chunks = list(all_chunks_by_id.values())
        write_jsonl_models(self.config.chunks_path, all_chunks)

        vector_store = FaissChunkStore(self.config.faiss_path, self.config.vector_metadata_path, self.embedding_client)
        if overwrite_vectors or not self.config.faiss_path.exists() or not self.config.vector_metadata_path.exists():
            vector_store.build(vector_chunks, dimension=self.config.embedding_dimension)
        else:
            vector_store.load()

        if overwrite_triples and self.config.triples_path.exists():
            self.config.triples_path.unlink()

        existing = _load_existing_triples(self.config.triples_path)
        completed_chunks = {triple.chunk_id for triple in existing}
        new_triples: list[TripleRecord] = []
        for chunk in tqdm(kg_chunks, desc="Extracting KGCaRe KG", unit="chunk"):
            if chunk.chunk_id in completed_chunks:
                continue
            _, _, _, parsed = run_three_step_kg_extraction(
                self.kg_llm,
                self.config.dataset,
                chunk.text,
                max_tokens=max_tokens,
                output_mode=kg_output_mode,
            )
            records = [
                TripleRecord(
                    triple_id=triple_id(self.config.dataset, self.config.index_name, chunk.chunk_id, triple),
                    dataset=self.config.dataset,
                    index_name=self.config.index_name,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    head=triple[0],
                    relation=triple[1],
                    tail=triple[2],
                    source_path=chunk.source_path,
                    evidence=find_evidence_sentence(chunk.text, triple),
                    metadata={"title": chunk.title, **chunk.metadata},
                )
                for triple in parsed[: self.config.max_triplets_per_chunk]
            ]
            if records:
                write_jsonl_models(self.config.triples_path, records, append=True)
                new_triples.extend(records)

        all_triples = _load_existing_triples(self.config.triples_path)
        graph = Neo4jTripleStore(self.config.neo4j, self.config.dataset, self.config.index_name)
        try:
            if reset_graph:
                graph.reset_namespace()
            graph.upsert_triples(all_triples)
            stats = graph.stats()
        finally:
            graph.close()

        manifest = IndexManifest(
            dataset=self.config.dataset,
            index_name=self.config.index_name,
            kg_model=self.config.kg_model,
            kg_output_mode=kg_output_mode,
            embedding_model=self.config.embedding_model,
            embedding_dimension=self.config.embedding_dimension,
            neo4j_uri=self.config.neo4j.uri,
            neo4j_database=self.config.neo4j.database,
            document_count=len({doc.doc_id for doc in vector_docs + kg_docs}),
            chunk_count=len(all_chunks),
            vector_count=vector_store.count,
            triple_count=len(all_triples),
            node_count=stats.get("nodes"),
            edge_count=stats.get("edges"),
            files={
                "chunks": str(self.config.chunks_path),
                "triples": str(self.config.triples_path),
                "faiss": str(self.config.faiss_path),
                "vector_metadata": str(self.config.vector_metadata_path),
            },
        )
        self.config.manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest
