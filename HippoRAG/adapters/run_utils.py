from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

from common import iter_jsonl, run_dir, write_json, write_jsonl

HIPPORAG_ROOT = Path(__file__).resolve().parents[1]
if str(HIPPORAG_ROOT) not in sys.path:
    sys.path.insert(0, str(HIPPORAG_ROOT))

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig


def load_corpus_docs(corpus_path: Path) -> list[str]:
    rows = list(iter_jsonl(corpus_path))
    docs: list[str] = []
    bar = tqdm(rows, desc="Loading corpus", unit="doc", dynamic_ncols=True)
    for row in bar:
        bar.set_postfix_str(row.get("title", row["id"])[:40])
        docs.append(f"{row['title']}\n{row['text']}")
    return docs


def load_query_rows(queries_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(queries_path), start=1):
        if limit is not None and idx > limit:
            break
        rows.append(row)
    return rows


def run_hipporag(
    *,
    dataset_key: str,
    provider: str,
    llm_model: str,
    llm_base_url: str | None,
    embedding_model: str,
    embedding_base_url: str | None,
    query_rows: list[dict[str, Any]],
    docs: list[str],
    query_builder: Callable[[dict[str, Any]], str],
    run_name: str,
    force_index_from_scratch: bool,
    force_openie_from_scratch: bool,
    retrieval_top_k: int,
    qa_top_k: int,
    max_new_tokens: int,
    temperature: float,
    ingest: bool,
    # Shared-index support: decouple the index dir from the QA run dir.
    # When set, the graph/VDB are stored here regardless of llm_model or run_name.
    index_dir: Path | None = None,
    # When True, build the index and exit without running QA.
    index_only: bool = False,
    # Override where predictions.jsonl is written (defaults to save_path).
    predictions_path_override: Path | None = None,
) -> tuple[Path, Path | None, Path]:
    qa_run_path = run_dir(dataset_name=dataset_key, provider=provider, model_name=llm_model, run_name=run_name)
    # Index lives in its own dir when specified, otherwise co-located with the QA run.
    save_path = index_dir or qa_run_path
    predictions_path = predictions_path_override or (qa_run_path / "predictions.jsonl")
    config_path = qa_run_path / "run_config.json"

    if llm_base_url and ("localhost" in llm_base_url or "127.0.0.1" in llm_base_url) and os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = "not_needed"

    global_config = BaseConfig(
        save_dir=str(save_path),
        dataset=dataset_key,
        llm_name=llm_model,
        llm_base_url=llm_base_url,
        embedding_model_name=embedding_model,
        embedding_base_url=embedding_base_url,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        retrieval_top_k=retrieval_top_k,
        qa_top_k=qa_top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        graph_type="facts_and_sim_passage_node_unidirectional",
    )

    write_json(
        config_path,
        {
            "dataset": dataset_key,
            "provider": provider,
            "llm_model": llm_model,
            "llm_base_url": llm_base_url,
            "embedding_model": embedding_model,
            "embedding_base_url": embedding_base_url,
            "run_name": run_name,
            "force_index_from_scratch": force_index_from_scratch,
            "force_openie_from_scratch": force_openie_from_scratch,
            "retrieval_top_k": retrieval_top_k,
            "qa_top_k": qa_top_k,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "num_docs": len(docs),
            "num_queries": len(query_rows),
            "ingest": ingest,
            "index_dir": str(save_path),
            "index_only": index_only,
        },
    )

    hipporag = HippoRAG(global_config=global_config)

    existing_doc_ids = hipporag.chunk_embedding_store.get_all_ids()
    should_index = ingest or force_index_from_scratch or len(existing_doc_ids) == 0
    if should_index:
        print(f"Indexing {len(docs)} documents into {save_path} ...")
        hipporag.index(docs=docs)
        print("Indexing complete.")

    if index_only:
        print(f"--index-only: skipping QA. Index stored at {save_path}")
        return save_path, None, config_path

    queries = [query_builder(row) for row in query_rows]

    print(f"Running QA on {len(queries)} queries ...")
    queries_solutions, all_response_message, all_metadata = hipporag.rag_qa(queries=queries)

    rows: list[dict[str, Any]] = []
    for row, query_solution, raw_response, metadata in zip(query_rows, queries_solutions, all_response_message, all_metadata):
        rows.append(
            {
                "id": row["id"],
                "question": row.get("question", ""),
                "model_query": query_solution.question,
                "question_type": row.get("question_type", ""),
                "reference_answers": row.get("reference_answers"),
                "reference_answer": row.get("reference_answer"),
                "raw_response": raw_response,
                "predicted_answer": query_solution.answer,
                "retrieved_docs": query_solution.docs,
                "metadata": metadata,
            }
        )

    write_jsonl(predictions_path, rows)
    return save_path, predictions_path, config_path
