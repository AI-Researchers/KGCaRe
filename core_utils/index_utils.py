import os
import logging
import faiss

from llama_index.core import (
    StorageContext, load_index_from_storage, VectorStoreIndex
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core import KnowledgeGraphIndex

from core_utils.multi_prompt_kg_index import MultiPromptKnowledgeGraphIndex  # your custom version
import core_utils.all_prompts as cs_prompts  # prompt templates

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Neo4j Setup
def setup_neo4j_graph_store():
    logger.info("Setting up Neo4j Graph Store...")
    graph_store = Neo4jGraphStore(
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        database=os.getenv("NEO4J_DATABASE", "neo4j")
    )
    logger.info("Neo4j Graph Store setup completed.")
    return graph_store


# Knowledge Graph Index
def create_or_load_kg_index(nodes, graph_store, index_type="KnowledgeGraphIndex",persist_dir="./storage/storage_graph_kg_dev_v3", index_id="dev_kg"):
    """
    Args:
        nodes: preprocessed text nodes
        graph_store: Neo4jGraphStore instance
        index_type: "multiprompt" or "standard"
    """
    logger.info(f"Creating or loading Knowledge Graph Index (type={index_type})...")

    graph_storage_context = StorageContext.from_defaults(graph_store=graph_store)

    if not os.path.exists(persist_dir):
        if index_type == "multiprompt_index":
            prompts = [
                PromptTemplate(cs_prompts.MLT_PROMPT_1),
                PromptTemplate(cs_prompts.MLT_PROMPT_2),
                PromptTemplate(cs_prompts.MLT_PROMPT_3),
            ]
            kg_index = MultiPromptKnowledgeGraphIndex(
                nodes,
                storage_context=graph_storage_context,
                max_triplets_per_chunk=30,
                show_progress=True,
                kg_triplet_extract_templates=prompts,
                include_embeddings=True,
                max_object_length=1000,
            )
        elif index_type == "KnowledgeGraphIndex":  # standard KnowledgeGraphIndex (for HybridContextQA)
            default_prompt = PromptTemplate(
                cs_prompts.DEFAULT_KG_TRIPLET_EXTRACT_TMPL_3,
                prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
            )
            kg_index = KnowledgeGraphIndex(
                nodes,
                storage_context=graph_storage_context,
                max_triplets_per_chunk=30,
                show_progress=True,
                kg_triplet_extract_template=default_prompt,
                include_embeddings=True,
                max_object_length=1000,
            )

            kg_index.set_index_id(index_id)
            kg_index.storage_context.persist(persist_dir=persist_dir)

    else:
        graph_storage_context = StorageContext.from_defaults(
            graph_store=graph_store,
            persist_dir=persist_dir
        )
        kg_index = load_index_from_storage(
            graph_storage_context, index_id=index_id
        )

    logger.info("Knowledge Graph Index created or loaded successfully.")
    return kg_index


# Faiss Vector Store
def setup_faiss_vector_store(dim=4096):
    logger.info("Setting up Faiss Vector Store...")
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    logger.info("Faiss Vector Store setup completed.")
    return vector_store


# Vector Index
def create_or_load_vector_index(nodes, vector_store,persist_dir="./storage/storage_vector_dev_v3", index_id="vector_index_dev"):
    logger.info("Creating or loading Vector Store Index...")

    if not os.path.exists(persist_dir):
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        vector_index.set_index_id(index_id)
        vector_index.storage_context.persist(persist_dir=persist_dir)

    else:
        vector_store = FaissVectorStore.from_persist_dir(persist_dir=persist_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=persist_dir
        )
        vector_index = load_index_from_storage(storage_context, index_id=index_id)

    logger.info("Vector Store Index created or loaded successfully.")
    return vector_index
