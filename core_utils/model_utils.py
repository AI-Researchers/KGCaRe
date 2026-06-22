import os
import logging
# from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.vllm import Vllm
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.cohere import Cohere
# from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.postprocessor import LLMRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank
from typing import Optional, Tuple
from llama_index.core.llms import LLM



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def load_llm_and_embeddings(using: str = "huggingface", llmmodel_name: str = None):
    """Load LLM and embedding model, and set global Settings."""
    logger.info("Loading LLM and Embeddings...")

    # Load LLM
    if using == "openai":
        key = os.getenv('OPENAI_API_KEY')
        llm = OpenAI(model=llmmodel_name, api_key=key)

    elif using == "ollama":
        llm = Ollama(
            base_url="http://127.0.0.1:11434",
            model=llmmodel_name,
            request_timeout=120.0,
        )

    elif using == "vllm":
        llm = Vllm(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            tensor_parallel_size=4,
            max_new_tokens=256,
        )

    elif using == "mistralai":
        api_key = os.getenv('MistralAI_API_KEY')
        llm = MistralAI(
            api_key=api_key,
            model=llmmodel_name,
            temperature=0.0,
        )

    elif using == "cohere":
        cohere_api_key = os.getenv("CO_API_KEY")
        llm = Cohere(
            model="command-r-plus",
            api_key=cohere_api_key,
            temperature=0.0,
        )

    # elif using == "openai_like":
    #     llm = OpenAILike(
    #         model="mistralai/Mistral-7B-Instruct-v0.1",
    #         api_base="http://0.0.0.0:8000/v1",
    #         api_key=os.getenv("OPENAI_API_KEY", "not_needed"),
    #         is_chat_model=True,
    #         max_tokens=128,
    #     )

    else:
        raise ValueError(f"Unknown LLM framework: {using}")

    # Load embedding model
    embedding_llm = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        trust_remote_code=True,
        device="cuda",
    )


    logger.info("LLM and Embeddings loaded successfully.")
    return llm, embedding_llm


def load_reranker(reranker_type: str, llm: Optional[LLM] = None):
    """
    Load a reranker module.
    
    Args:
        reranker_type (str): 'none', 'cohere', or 'llm'
        llm (LLM, optional): LLM instance needed for LLMRerank

    Returns:
        reranker instance or None
    """
    if reranker_type == "none":
        return None
    elif reranker_type == "cohere":
        return CohereRerank(
            model='rerank-english-v3.0',
            api_key=os.environ.get("CO_API_KEY"),
            top_n=2
        )
    elif reranker_type == "llm":
        if llm is None:
            raise ValueError("LLM must be provided for LLM reranker")
        return LLMRerank(llm=llm, choice_batch_size=10, top_n=2)
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
