#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, traceback
from pathlib import Path
from typing import List
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core_utils")))


# llama-index imports
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import KGTableRetriever, VectorIndexRetriever
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai import OpenAI

# --- KGRetrieverToGTraversal_Wikidata (Version 7.0 - Final & Resilient) ---
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.core.prompts.default_prompts import (
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
from llama_index.core.settings import Settings
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms.llm import LLM
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from kg_retriever import KGCaRe

from typing import List, Dict, Optional, Set, Tuple
import re
from SPARQLWrapper import SPARQLWrapper, JSON
import time

# ----------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Retrieve vector & KG context for each HotpotQA question.")
    p.add_argument("--input", type=Path, required=True, help="HotpotQA JSON input file")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL file with contexts")
    p.add_argument("--vector-storage", type=Path, default=Path("./storage/dev_wiki_articles_500_neo4j_faiss_kg_v3"))
    p.add_argument("--kg-storage", type=Path, default=Path("./storage/dev_wiki_articles_500_neo4j_faiss_kg_v2"))
    p.add_argument("--framework", type=str, default='openai', choices=['openai', 'openai-like'],)
    p.add_argument("--model", type=str, default='gpt-3.5-turbo', help="LLM model to use")
    p.add_argument("--similarity-top-k", type=int, default=10)
    p.add_argument("--kg-top-k", type=int, default=30)
    return p

# ----------------------------------------------------------------------
# Retrieval loaders
# ----------------------------------------------------------------------
def load_vector_retriever(storage_dir: Path, embed_model, similarity_top_k: int):
    try:
        vector_store = FaissVectorStore.from_persist_dir(str(storage_dir))
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(storage_dir))
        vector_index = load_index_from_storage(storage_context=storage_ctx,index_id="vector_index_dev")
        return VectorIndexRetriever(embed_model=embed_model, index=vector_index, similarity_top_k=similarity_top_k)
    except Exception:
        print(f"⚠️ Failed to load vector retriever from {storage_dir}")
        traceback.print_exc()
        exit(1)
        return None

def load_kg_retriever(storage_dir: Path):
    try:
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        graph_store = Neo4jGraphStore(
            username=username,
            password=password,
            url=url,
            database=database,
        )
        kg_storage_ctx = StorageContext.from_defaults(graph_store=graph_store, persist_dir=str(storage_dir))
        kg_index = load_index_from_storage(kg_storage_ctx,index_id="Kg_Index_dev")
        return KGRetrieverToGTraversal_final(index=kg_index, graph_store=graph_store,  max_depth=3, max_entities=20)
    except Exception:
        print(f"⚠️ Failed to load KG retriever from {storage_dir}")
        traceback.print_exc()
        exit(1)
        return None

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(args):
    
    # Setup output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    if args.framework == 'openai-like':
            # Setup LLM and Embedding Model
        llm = OpenAILike(
            model=args.model,                                     #"mistralai/Mixtral-8x7B-Instruct-v0.1","mistralai/Mistral-7B-Instruct-v0.3"
            api_base=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "not_needed"),
            is_chat_model=True,
            max_tokens=128,
        )
    elif args.framework == 'openai':
        llm = OpenAI(
            model=args.model,
        )
    
    Settings.llm = llm

    # Load embed model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5",device="cuda:2")
    Settings.embed_model = embed_model

    vector_retriever = None
    kg_retriever = None
    
    print(f"📦 Loading vector retriever from {args.vector_storage}...")
    vector_retriever = load_vector_retriever(args.vector_storage, embed_model, args.similarity_top_k)

    print(f"📦 Loading KG retriever from {args.kg_storage}...")
    kg_retriever = load_kg_retriever(args.kg_storage)

    # Load dataset
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except Exception:
        print(f"❌ Failed to load input file: {args.input}")
        traceback.print_exc()
        return

    # Resume support
    completed_ids = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    completed_ids.add(obj["_id"])
                except Exception:
                    continue
        print(f"▶️ Resuming: {len(completed_ids)} already done")

    # Open output in append mode
    with args.output.open("a", encoding="utf-8") as fout:
        for item in tqdm(data, desc="Retrieving", unit="q"):
            qid = item.get("_id")
            question = item.get("question")
            # entities = item.get("entities")
            entities = None
            if not qid or not question:
                print("⚠️ Skipping invalid item with no _id or question.")
                continue
            if qid in completed_ids:
                continue

            try:
                vector_context = ""
                kg_context = ""

                try:
                    nodes = vector_retriever.retrieve(question)
                    vector_context = "\n\n".join(node.get_content() for node in nodes)
                except Exception:
                    print(f"⚠️ Vector retrieval failed for id {qid}")
                    traceback.print_exc()
 
                try:
                    knodes = kg_retriever._retrieve(question,entities)
                    if knodes:
                        kg_context = "\n".join(node.get_content() for node in knodes)
                except Exception:
                    print(f"⚠️ KG retrieval failed for id {qid}")
                    traceback.print_exc()

                record = {
                    "_id": qid,
                    "question": question,
                    "answer": item.get("answer", ""),
                    "vector_context": vector_context,
                    "kg_context": kg_context
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception:
                print(f"❌ Unexpected error while processing id {qid}")
                traceback.print_exc()
                continue

    print(f"✅ Retrieval complete. JSONL saved to {args.output}")

# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)
