import os
import json
import torch
import logging
import sys
import itertools
import math
import collections
import string
import re
import copy
import argparse
import faiss
import nest_asyncio
from tqdm import tqdm
from glob import glob
from typing import Optional, Callable, Any
import os
import json
import time
from datetime import datetime

from llama_index.core import (
    Settings, VectorStoreIndex, SimpleDirectoryReader, KnowledgeGraphIndex, 
    load_index_from_storage, StorageContext, QueryBundle
)
# from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter, HTMLNodeParser
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import LLMRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import get_response_synthesizer
from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.vllm import Vllm
# from llama_index.llms.openai_like import OpenAILike
from llama_index.core.schema import NodeWithScore , TextNode
from llama_index.llms.openai import OpenAI



from custom_retriever import CustomRetriever , CustomRetrieverToGTraversal , CustomRetrieverToG
from QA_classifier import QuestionTypeClassifier
from evaluation import evaluate , load_answers, compute_metrics
from bert_score import BERTScorer            
from evaluation_bert import evaluate_meteor_bert ,compute_meteor_bert_metrics
from document_reader import HTMLDocsReader
from few_shot_selection import select_random_few_shots
import all_prompts as cs_prompts
import QA_classifier as qtc
from multi_prompt_kg_index import MultiPromptKnowledgeGraphIndex
from kg_retriever import KGRetrieverToG , KGRetrieverToGTraversal, KGRetrieverToGTraversal_final

nest_asyncio.apply()

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

# Function to load LLM and Embeddings
def load_llm_and_embeddings(using="huggingface", llmmodel_name=None):
    logger.info("Loading LLM and Embeddings...")
    
    if using == "openai":
        key = os.getenv('OPENAI_API_KEY')
        llm = OpenAI(model=llmmodel_name, api_key=key)
    elif using == "ollama":
        llm = Ollama(base_url='http://127.0.0.1:11434',model=llmmodel_name, request_timeout=120.0,)
    # elif  using == "huggingface":
    #     llm = HuggingFaceLLM(
    #         context_window=2048,
    #         max_new_tokens=256,
    #         generate_kwargs={"temperature": 0, "do_sample": False},
    #         tokenizer_name="mistralai/Mixtral-8x7B-v0.1",
    #         model_name="mistralai/Mixtral-8x7B-v0.1",
    #         device_map="auto",
    #         model_kwargs={"torch_dtype": torch.float16}
    #     )
    elif using == "vllm":
        llm = Vllm(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            tensor_parallel_size=4,
            max_new_tokens=256,
            # vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
        )
    elif using == "mistralai":
        api_key = os.getenv('MistralAI_API_KEY')
        llm = MistralAI(api_key=api_key,model=llmmodel_name,temperature=0.0) #open-mistral-7b open-mixtral-8x7b open-mistral-nemo
    elif using == "cohere":
        cohere_api_key = os.getenv('CO_API_KEY')
        llm = Cohere(model="command-r-plus", api_key=cohere_api_key, temperature=0.0)
    # elif using == "openai_like":
        # llm = OpenAILike(model="mistralai/Mixtral-8x7B-v0.1", api_base="http://0.0.0.0:8000/v1", api_key="token-abc123")
        
        
    embedding_llm = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5",trust_remote_code=True,device ='cuda') # nvidia/NV-Embed-v2 , BAAI/bge-large-en-v1.5
    
        
    Settings.llm = llm
    Settings.embed_model = embedding_llm
    
    logger.info("LLM and Embeddings loaded successfully.")
    
    return llm, embedding_llm

# Function to load HTML documents
def load_html_docs(filepath):
    logger.info(f"Loading HTML documents from {filepath}...")
    loader = SimpleDirectoryReader(
        input_dir=filepath, 
        exclude=["*.rst", "*.ipynb", "*.py", "*.bat", "*.png", "*.jpg", "*.jpeg", "*.csv", "*.html", "*.js", "*.css", "*.pdf", "*.json"],
        file_extractor={".txt": HTMLDocsReader(tags=["h1"])},
        recursive=True
    )
    nodes = loader.load_data()
    logger.info("HTML documents loaded successfully.")
    return nodes

# Function to modify metadata
def modify_metadata(nodes):
    logger.info("Modifying metadata for nodes...")
    text_template = "Content Metadata:\n{metadata_str}\n\nContent:\n{content}"
    metadata_template = "{key}: {value},"

    for doc in nodes:
        doc.text_template = text_template
        doc.metadata_template = metadata_template
        doc.excluded_llm_metadata_keys = [ 'File Name','file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date','file_path','Content Type']
        doc.excluded_embed_metadata_keys = [ 'File Name','file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date', 'file_path','Content Type']
    logger.info("Metadata modification completed.")
    return nodes

# Function to setup Neo4j Graph Store
def setup_neo4j_graph_store():
    logger.info("Setting up Neo4j Graph Store...")
    username = "neo4j"
    password = "P@ssw0rd"
    url = "bolt://localhost:7688"
    database = "neo4j"
    graph_store = Neo4jGraphStore(
        username=username,
        password=password,
        url=url,
        database=database,
    )
    logger.info("Neo4j Graph Store setup completed.")
    return graph_store

# Function to create or load Knowledge Graph Index
def create_or_load_kg_index(nodes, graph_store):
    logger.info("Creating or loading Knowledge Graph Index...")
    graph_storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
#     DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
#     cs_prompts.DEFAULT_KG_TRIPLET_EXTRACT_TMPL_3, # change prompt template
#     prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
# )
    
#     PERSIST_DIR = "./storage/storage_graph_kg_dev_manual_kg_all_v2"                                       # KG : change persist dir
    
    
#     if not os.path.exists(PERSIST_DIR):
#         kg_index  = KnowledgeGraphIndex(nodes,
#                         storage_context=graph_storage_context, 
#                         max_triplets_per_chunk=30,
#                         show_progress=True,
#                         kg_triplet_extract_template=DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
#                         include_embeddings = True
#                         max_object_length = 1000,
#                     )
#         kg_index.set_index_id("dev_manual_kg")                                                          # KG : change index id
#         kg_index.storage_context.persist(persist_dir=PERSIST_DIR)
#     else:
#         graph_storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=PERSIST_DIR)
#         kg_index = load_index_from_storage(graph_storage_context, index_id="dev_manual_kg")             # KG : change index id
#     logger.info("Knowledge Graph Index created or loaded successfully.")
#     return kg_index

    MLT_PROMPT_1 = PromptTemplate(cs_prompts.MLT_PROMPT_1)
    MLT_PROMPT_2 = PromptTemplate(cs_prompts.MLT_PROMPT_2)
    MLT_PROMPT_3 = PromptTemplate(cs_prompts.MLT_PROMPT_3)
    MLT_PROMPT = [MLT_PROMPT_1, MLT_PROMPT_2, MLT_PROMPT_3]
    
    PERSIST_DIR = "./storage/storage_graph_kg_dev_v3"                                       # KG : change persist dir
    
    if not os.path.exists(PERSIST_DIR):
        kg_index  = MultiPromptKnowledgeGraphIndex(nodes,
                        storage_context=graph_storage_context, 
                        max_triplets_per_chunk=30,
                        show_progress=True,
                        kg_triplet_extract_templates=MLT_PROMPT,
                        include_embeddings = True,
                        max_object_length = 1000,
                    )
        kg_index.set_index_id("dev_kg")                                                          # KG : change index id
        kg_index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        graph_storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=PERSIST_DIR)
        kg_index = load_index_from_storage(graph_storage_context, index_id="dev_kg")             # KG : change index id
    logger.info("Knowledge Graph Index created or loaded successfully.")
    return kg_index


# Function to setup Faiss Vector Store
def setup_faiss_vector_store():
    logger.info("Setting up Faiss Vector Store...")
    d = 4096                                                # VECTOR : change dimension  1024 for bgelarge
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    logger.info("Faiss Vector Store setup completed.")
    return vector_store

# Function to create or load Vector Store Index
def create_or_load_vector_index(nodes, vector_store):
    logger.info("Creating or loading Vector Store Index...")
    vector_storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    PERSIST_DIR = "./storage/storage_vector_dev_v3"                              # VECTOR : change persist dir
    
    if not os.path.exists(PERSIST_DIR):
        vector_index = VectorStoreIndex(nodes, storage_context=vector_storage_context)
        vector_index.set_index_id("vector_index_dev")                         # VECTOR : change index id      
        vector_index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        vector_store = FaissVectorStore.from_persist_dir(persist_dir=PERSIST_DIR)
        vector_storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
        vector_index = load_index_from_storage(storage_context=vector_storage_context, index_id="vector_index_dev")     # VECTOR : change persist dir
    logger.info("Vector Store Index created or loaded successfully.")
    return vector_index


# Custom Query Engine class
class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""
    
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, data_FS,query_str: str,q_type):
        logger.info(f"Question Type: {q_type}")

        
        # Load template based on question type
        if q_type == "yes/no":
            qa_prompt = cs_prompts.yes_no_qa_s4_prompt
        elif q_type == "yes/no_conditional":
            qa_prompt = cs_prompts.yes_no_con_qa_s4_prompt
        elif q_type == "span":
            qa_prompt = cs_prompts.span_qa_s4_prompt
        elif q_type == "span_conditional":
            qa_prompt = cs_prompts.span_con_qa_s4_prompt
        else:
            qa_prompt = cs_prompts.span_qa_s4_prompt
        
        
        query_str = QueryBundle(query_str)
        
        if args.index == "vector_index":
            nodes  = self.retriever._retrieve(query_str)
            Text_context_str = "\n\n".join([n.node.get_content() for n in nodes ])
            context_str = Text_context_str
        elif args.index == "kg_index":
            nodes  = self.retriever._retrieve(query_str)
            kg_context_str = "\n\n".join([n.node.get_content() for n in nodes ])
            context_str = "\n\n Knowledge Triples: "+ kg_context_str
        elif args.index == "hybrid_index":
            nodes , dict_retrieve_nodes = self.retriever._retrieve(query_str)
            Text_context_str = "\n\n".join([n.node.get_content() for n in dict_retrieve_nodes['vector']])
            kg_context_str = "\n".join([n.node.get_content() for n in dict_retrieve_nodes['kg']])
            context_str = Text_context_str + "\n\n Knowledge Triples: " + kg_context_str
        
        context_str_node = [NodeWithScore(node=TextNode(text=str(context_str)))]
        
        ####---------WITH RANDOM SHOTS-------------####
        # Update the prompt with few shots examples
        # few_shot_examples = select_random_few_shots(data_FS, q_type, args.num_shots)
        
        # prompt_ = qa_prompt.partial_format(few_shot_examples=few_shot_examples,context_str=context_str)
        # prompt_out = qa_prompt.format(few_shot_examples=few_shot_examples,query_str=query_str,context_str=context_str)
        ####---------WITH STATIC SHOTS-------------####
        prompt_ = qa_prompt.partial_format(context_str=context_str)
        prompt_out = qa_prompt.format(query_str=query_str,context_str=context_str)
        # print(prompt_out)
        ####----------------------####
        self.response_synthesizer.update_prompts({"text_qa_template": prompt_}) #text_qa_template  summary_template
        
        response_obj = self.response_synthesizer.synthesize(query_str, context_str_node)
        
        return response_obj , prompt_out

def format_prediction(prediction, qtype):
    answer = prediction.encode('utf-8').decode('unicode_escape')
    conditions = []
    
    print(prediction)
    if qtype == "yes/no" or qtype == "span":
        try:
            answer = prediction.split("Answer: ")[1].strip()
        except IndexError:
            answer = prediction
            
        if qtype == "yes/no" and 'yes' in answer.lower().split(' '):
            formatted_output = [["yes", []]]
        elif qtype == "yes/no" and 'no' in answer.lower().split(' '):
            formatted_output = [["no", []]]
        else:
            formatted_output = [[answer, []]]
    else:
        try:
            conditions_text = answer.split("Conditions: ")[1].strip()
            conditions = conditions_text.split("\n")
            answer = answer.split("Conditions:")[0].strip().split("Answer:")[1].strip()
            
            if qtype == "yes/no" and 'yes' in answer.lower().split(' '):
                formatted_output = [["yes", conditions]]
            elif qtype == "yes/no" and 'no' in answer.lower().split(' '):
                formatted_output = [["no", conditions]]
            else:
                formatted_output = [[answer, conditions]]
        except IndexError:
            conditions = []
            formatted_output = [
                [answer,conditions]
            ]

    return formatted_output

def parse_arguments():
    # command-line flags are defined here.
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--llm_framework",
        dest="llm_framework",
        type=str,
        choices=['ollama', 'huggingface', 'mistralai', 'openai', 'cohere', 'vllm', 'openai_like', 'openai'],
        default='ollama',
        help="Framework or package to load the model",
    )
    parser.add_argument(
        "--llm_model",
        dest="llm_model",
        type=str,
        default='llama3',
        help="name of the model to be used",
    )
    parser.add_argument(
        "--emb_model",
        dest="emb_model",
        type=str,
        default=None,
        help="emb model to be used",
    )
    parser.add_argument(
        "--index",
        dest="index",
        type=str,
        choices=['kg_index', 'vector_index', 'hybrid_index'],
        default='hybrid_index',
        help="index to be used for retrieval", 
    )
    parser.add_argument(
        "--doc_path",
        dest="doc_path",
        type=str,
        default='/home/simsam/conditionalqa_rag_pipeline/data/docs_dev',
        help="path to the documents to be indexed", 
    )
    parser.add_argument(
        "--num_shots",
        dest="num_shots",
        type=int,
        default=2,
        help="use few shot examples", 
    )
    parser.add_argument(
        "--ref_file",
        dest="ref_file",
        type=str,
        default='/home/simsam/conditionalqa_rag_pipeline/data/dev.json', #/yes_no_questions.json /dev_10.json #/span_questions.json #span_conditional_questions.json #yes_no_conditional_questions.json
        help="use conditions", 
    )
    
    parser.add_argument(
        "--retry_skipped",
        action="store_true",
        default=False,
        help="Retry only the skipped IDs stored in skipped_ids.json"
    )
    
    return parser.parse_args()

def load_processed_ids(jsonl_path):
    if not os.path.exists(jsonl_path):
        return set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return {json.loads(line.strip())["id"] for line in f if line.strip()}

if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    # Load LLM and Embeddings
    llm, embedding_llm = load_llm_and_embeddings(using=args.llm_framework, llmmodel_name=args.llm_model)

    # Load HTML documents
    nodes = load_html_docs(args.doc_path)
    nodes = modify_metadata(nodes)

    # Setup stores
    graph_store = setup_neo4j_graph_store()
    kg_index = create_or_load_kg_index(nodes, graph_store)
    vector_store = setup_faiss_vector_store()
    vector_index = create_or_load_vector_index(nodes, vector_store)

    # Synthesizer and retrievers
    response_synthesizer = get_response_synthesizer(response_mode="simple_summarize", verbose=True)
    kg_keyword_retriever = KGRetrieverToGTraversal_final(index=kg_index, graph_store=graph_store, llm=llm, max_depth=3, max_entities=20)
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5, verbose=False)

    if args.index == "kg_index":
        retriever = kg_keyword_retriever
    elif args.index == "vector_index":
        retriever = vector_retriever
    elif args.index == "hybrid_index":
        retriever = CustomRetrieverToGTraversal(vector_retriever, kg_keyword_retriever)

    custom_query_engine = RAGQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    # Output paths
    output_path = os.path.join("outputs_kgtogt/dev_kg_v3/ConditionalQA_test/", args.llm_model, args.index, f"shots_{args.num_shots}")
    os.makedirs(output_path, exist_ok=True)
    num_runs = len(glob(os.path.join(output_path, "*")))
    output_path = os.path.join(output_path, f"run_{num_runs}")
    os.makedirs(output_path, exist_ok=True)
    print(f"Output path: {output_path}")

    output_file_path = os.path.join(output_path, "output.jsonl")
    skipped_path = os.path.join(output_path, "skipped_ids.json")

    # Load data
    with open(args.ref_file, 'r', encoding='utf-8') as f:
        evaluation_data = json.load(f)
    with open('/home/simsam/conditionalqa_rag_pipeline/data/train_FS_sampled_triples.json', 'r') as f:
        data_FS = json.load(f)

    # Filter if retry_skipped
    if getattr(args, "retry_skipped", False):
        if not os.path.exists(skipped_path):
            print(f"[ERROR] --retry_skipped is set but {skipped_path} does not exist.")
            exit(1)
        with open(skipped_path, "r") as f:
            skipped_ids_to_retry = set(json.load(f))
        evaluation_data = [d for d in evaluation_data if d["id"] in skipped_ids_to_retry]
        print(f"[INFO] Retrying {len(evaluation_data)} skipped questions.")

    processed_ids = load_processed_ids(output_file_path)
    skipped_ids = []
    bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")

    for data in tqdm(evaluation_data):
        if data["id"] in processed_ids or data.get("not_answerable", False):
            continue

        scenario = data['scenario']
        question = data['question']
        cq = f"{scenario} {question}"
        question_type = qtc.classify_single_question(data['answers'])

        # Retry logic for custom_query
        success = False
        last_exception = None
        for attempt in range(3):
            try:
                response, prompt_ = custom_query_engine.custom_query(data_FS, cq, question_type)
                success = True
                break
            except Exception as e:
                last_exception = e
                print(f"[ERROR] custom_query failed for ID: {data['id']} (Attempt {attempt + 1}/3): {e}")
                time.sleep(5)

        if not success:
            print(f"[FAILED] Skipping ID: {data['id']}")
            skipped_ids.append(data['id'])
            fallback = {
                "id": data['id'],
                "Prompt": None,
                "Question": cq,
                "Question_Type": question_type,
                "answers": None,
                "Actual_Answer": data['answers'],
                "Score": None,
                "Error": str(e),
            }
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
            continue

        try:
            formatted_response = format_prediction(response.response, question_type)
            em, conditional_em, f1, conditional_f1 = compute_metrics(formatted_response, data['answers'])
            em, conditional_em, f1, conditional_f1 = compute_meteor_bert_metrics(formatted_response, data['answers'], bert_scorer)

            result = {
                "id": data['id'],
                "Prompt": prompt_,
                "Question": cq,
                "Question_Type": question_type,
                "answers": formatted_response,
                "Actual_Answer": data['answers'],
                "Score": {
                    'EM': em,
                    'Conditional_EM': conditional_em,
                    'F1': f1,
                    'Conditional_F1': conditional_f1
                },
            }
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[ERROR] Scoring failed for ID: {data['id']} - {str(e)}")
            skipped_ids.append(data['id'])
            
            fallback = {
                "id": data['id'],
                "Prompt": prompt_,
                "Question": cq,
                "Question_Type": question_type,
                "answers": None,
                "Actual_Answer": data['answers'],
                "Score": None,
                "Error": str(e),
            }
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fallback, ensure_ascii=False) + "\\n")

    if skipped_ids:
        with open(skipped_path, "w", encoding="utf-8") as f:
            json.dump(skipped_ids, f, indent=2)
        print(f"[INFO] Skipped {len(skipped_ids)} IDs. Saved to: {skipped_path}")
    elif os.path.exists(skipped_path):
        os.remove(skipped_path)
        print(f"[INFO] All questions processed. Removed old skipped_ids.json")