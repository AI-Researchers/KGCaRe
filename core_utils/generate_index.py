import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import faiss

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    KnowledgeGraphIndex,
    StorageContext,
    Settings,
)
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.vllm import Vllm
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts.base import PromptTemplate
from multi_prompt_kg_index import MultiPromptKnowledgeGraphIndex
import all_prompts as cs_prompts
from llama_index.core.prompts.base import PromptTemplate


# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# -------------------------------
# Utility to modify metadata
# -------------------------------
def modify_metadata(nodes):
    text_template = "Content:\n{content}"
    metadata_template = "{key}: {value},"

    for doc in nodes:
        doc.text_template = text_template
        doc.metadata_template = metadata_template
        doc.excluded_llm_metadata_keys = [
            'File Name','file_type','file_size','creation_date',
            'last_modified_date','last_accessed_date','file_path','Content Type'
        ]
        doc.excluded_embed_metadata_keys = [
            'File Name','file_type','file_size','creation_date',
            'last_modified_date','last_accessed_date','file_path','Content Type'
        ]
    return nodes

# -------------------------------
# Prompt formatters for vLLM
# -------------------------------
# def messages_to_prompt(messages):
#     prompt = "\n".join([f"<{msg['role']}>{msg['content']}" for msg in messages])
#     return f"<s>[INST] {prompt} [/INST] </s>\n"

# def completion_to_prompt(completion):
#     return f"<s>[INST] {completion} [/INST] </s>\n"

# # -------------------------------
# # LLM
# # -------------------------------
# API_BASE = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
# API_KEY = os.getenv("OPENAI_API_KEY", "not_needed")
    
# llm = OpenAILike(
#     model="mistralai/Mixtral-8x7B-Instruct-v0.1",
#     api_base=API_BASE,
#     api_key=API_KEY,
#     is_chat_model=True,
#     max_tokens=512,
# )

llm = OpenAI(
    model="gpt-4o",
)

Settings.llm = llm

KG_INDEX = "kg_v1"  # or "kg_v1" for legacy

# -------------------------------
# Embedding model
# -------------------------------
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",
)
Settings.embed_model = embed_model

# -------------------------------
# Graph Store (Neo4j)
# -------------------------------
graph_store = Neo4jGraphStore(
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", ""),
    url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    database=os.getenv("NEO4J_DATABASE", "neo4j")
)

# -------------------------------
# Vector Store (Faiss)
# -------------------------------
d = 1024
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# -------------------------------
# Storage context combining both
# -------------------------------
storage_context = StorageContext.from_defaults(
    graph_store=graph_store,
    vector_store=vector_store,
)

# -------------------------------
# Load documents
# -------------------------------
repo_root = Path(__file__).resolve().parents[1]
docs_dir = repo_root / "data" / "wiki_articles_supported_500"
documents_kg = SimpleDirectoryReader(str(docs_dir), filename_as_id=True).load_data()
documents_kg = modify_metadata(documents_kg)

docs_dir = repo_root / "data" / "wiki_articles_500"
documents_vector = SimpleDirectoryReader(str(docs_dir), filename_as_id=True).load_data()
documents_vector = modify_metadata(documents_vector)

# -------------------------------
# Build KG Index
# -------------------------------
vector_index = VectorStoreIndex(documents_vector, storage_context=storage_context)


# -------------------------------
# Build KG Index
# -------------------------------
if KG_INDEX == "kg_v2":
    MLT_PROMPT_1 = PromptTemplate(cs_prompts.MLT_PROMPT_1)
    MLT_PROMPT_2 = PromptTemplate(cs_prompts.MLT_PROMPT_2)
    MLT_PROMPT_3 = PromptTemplate(cs_prompts.MLT_PROMPT_3)
    MLT_PROMPT = [MLT_PROMPT_1, MLT_PROMPT_2, MLT_PROMPT_3]
    
    graph_index  = MultiPromptKnowledgeGraphIndex(documents_kg,
                storage_context=storage_context, 
                max_triplets_per_chunk=30,
                show_progress=True,
                kg_triplet_extract_templates=MLT_PROMPT,
                include_embeddings = True,
                max_object_length = 1000,
            )
elif KG_INDEX == "kg_v1":
    graph_index = KnowledgeGraphIndex.from_documents(
        documents_kg,
        max_triplets_per_chunk=20,
        storage_context=storage_context,
        include_embeddings=True,
        show_progress=True,
    )

# -------------------------------
# Persist
# -------------------------------
PERSIST_DIR = "./storage/dev_wiki_articles_500_neo4j_faiss_kg_v2"  # change as needed
print(f"Saving index to {PERSIST_DIR} …")
Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
graph_index.set_index_id("Kg_Index_dev")                     # GRAPH : change index id
graph_index.storage_context.persist(persist_dir=PERSIST_DIR)
vector_index.set_index_id("vector_index_dev")                         # VECTOR : change index id      
vector_index.storage_context.persist(persist_dir=PERSIST_DIR)
print("✅ Indexing complete with Neo4j + Faiss!")
