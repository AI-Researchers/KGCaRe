# KGCaRe/conditionalqa/run.py

import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import json
import time
import argparse
from glob import glob
from tqdm import tqdm
from bert_score import BERTScorer

from core_utils.model_utils import load_llm_and_embeddings, load_reranker
from core_utils.model_utils import load_llm_and_embeddings
from core_utils.loaders import load_html_docs, modify_metadata
from core_utils.index_utils import (
    setup_faiss_vector_store,
    setup_neo4j_graph_store,
    create_or_load_vector_index,
    create_or_load_kg_index,
)
from core_utils.qa_classifier import classify_single_question as qtc
from core_utils.engine import RAGQueryEngine
from core_utils.format_utils import format_prediction , load_processed_ids
from core_utils.conditionalqa_eval import compute_metrics as default_compute_metrics
from core_utils.conditionalqa_eval_bert_score import compute_meteor_bert_metrics, BERTScorer
from core_utils.hybrid_context_retriever import HybridRetriever

from llama_index.core import get_response_synthesizer , KnowledgeGraphIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever, KGTableRetriever

from core_utils.kg_tripple_retrievers import KGCaRe
from core_utils.hybrid_context_retriever import HybridRetrieverKGCaRe

# --------------------------
# Settings / Default Paths
# --------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC_PATH = str(REPO_ROOT / "data" / "docs_dev")                      # Directory containing HTML documents to index
DEFAULT_REF_FILE = str(REPO_ROOT / "data" / "dev.json")                      # File with questions and answers to evaluate
KG_PERSIST_DIR = str(REPO_ROOT / "storage" / "storage_graph_kg_dev_v3")  # Directory to persist KG index
KG_INDEX_ID = "dev_kg"
VECTOR_PERSIST_DIR = str(REPO_ROOT / "storage" / "storage_vector_dev_v3")  # Directory to persist vector index
VECTOR_INDEX_ID = "vector_index_dev"                    # Directory to persist vector index
DEFAULT_EVAL_METHOD = "standard"                        # Choose from ['standard', 'bert']
DEFAULT_OUTPUT_ROOT = "outputs_final/dev_kg_v3/ConditionalQA_test"  # Root output directory
DEFAULT_NUM_SHOTS = 4                                   # Number of few-shot examples
DEFAULT_MODEL = "gpt-4o"                                # Default LLM model
DEFAULT_LLM_FRAMEWORK = "openai"                        # Framework to load the model
DEFAULT_INDEX_TYPE = "hybrid_index"                     # Index type
DEFAULT_RERANKER_TYPE = "cohere"                        # Choose from ['none', 'cohere', 'llm']
DEFAULT_EMB_MODEL = "BAAI/bge-large-en-v1.5"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_framework", type=str, choices=['openai', 'ollama', 'vllm', 'cohere', 'mistralai', 'openai_like'], default='ollama')
    parser.add_argument("--llm_model", type=str, default='llama3')
    parser.add_argument("--emb_model", type=str, default=DEFAULT_EMB_MODEL)
    parser.add_argument("--index", type=str, choices=['kg_index', 'vector_index', 'hybrid_index','no_index'], default=DEFAULT_INDEX_TYPE)
    parser.add_argument("--doc_path", type=str, default=DEFAULT_DOC_PATH)
    parser.add_argument("--ref_file", type=str, default=DEFAULT_REF_FILE)
    parser.add_argument("--num_shots", type=int, default=DEFAULT_NUM_SHOTS)
    parser.add_argument("--retry_skipped", action="store_true", default=False)
    parser.add_argument("--eval_method", type=str, choices=["standard", "bert"], default=DEFAULT_EVAL_METHOD)
    parser.add_argument("--reranker", type=str, choices=["none", "cohere", "llm"], default=DEFAULT_RERANKER_TYPE)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Load LLM + Embeddings + Reranker
    llm, embedding_llm = load_llm_and_embeddings(using=args.llm_framework, llmmodel_name=args.llm_model)
    reranker = load_reranker(args.reranker, llm=llm)
    Settings.embed_model = embedding_llm
    Settings.llm = llm
    
    # Load Documents
    nodes = load_html_docs(args.doc_path)
    nodes = modify_metadata(nodes)

    # Setup indexes
    graph_store = setup_neo4j_graph_store()
    kg_index = create_or_load_kg_index(nodes, graph_store, index_type='multiprompt_index', persist_dir=KG_PERSIST_DIR, index_id=KG_INDEX_ID)
    vector_store = setup_faiss_vector_store()
    vector_index = create_or_load_vector_index(nodes, vector_store,persist_dir=VECTOR_PERSIST_DIR, index_id=VECTOR_INDEX_ID)


    # Setup retrievers
    if args.index != 'no_index':
        kg_retriever = KGCaRe(index=kg_index, graph_store=graph_store, llm=llm, max_depth=3, max_entities=20)
        vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10, verbose=False)

        if args.index == "kg_index":
            retriever = kg_retriever
        elif args.index == "vector_index":
            retriever = vector_retriever
        elif args.index == "hybrid_index":
            retriever = HybridRetrieverKGCaRe(vector_retriever=vector_retriever, kg_retriever = kg_retriever, reranker=reranker)
    else:
        retriever = None

    # Setup RAG engine
    response_synthesizer = get_response_synthesizer(response_mode="simple_summarize", verbose=True)
    custom_query_engine = RAGQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    # Setup output directory
    output_path = os.path.join(DEFAULT_OUTPUT_ROOT, args.llm_model, args.index, f"shots_{args.num_shots}")
    os.makedirs(output_path, exist_ok=True)
    num_runs = len(glob(os.path.join(output_path, "*")))
    output_path = os.path.join(output_path, f"run_{num_runs}")
    os.makedirs(output_path, exist_ok=True)

    print(f"Output path: {output_path}")
    output_file_path = os.path.join(output_path, "output.jsonl")
    skipped_path = os.path.join(output_path, "skipped_ids.json")
    results_path = os.path.join(output_path, "results.json")

    # Load evaluation data
    with open(args.ref_file, 'r', encoding='utf-8') as f:
        evaluation_data = json.load(f)

    if args.retry_skipped:
        if not os.path.exists(skipped_path):
            print(f"[ERROR] --retry_skipped is set but {skipped_path} does not exist.")
            exit(1)
        with open(skipped_path, "r") as f:
            skipped_ids_to_retry = set(json.load(f))
        evaluation_data = [d for d in evaluation_data if d["id"] in skipped_ids_to_retry]
        print(f"[INFO] Retrying {len(evaluation_data)} skipped questions.")

    processed_ids = load_processed_ids(output_file_path)
    skipped_ids = []
    scores = []
    scorer_fn = default_compute_metrics
    if args.eval_method == "bert":
        scorer_fn = compute_meteor_bert_metrics
        bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")

    for data in tqdm(evaluation_data):
        if data["id"] in processed_ids or data.get("not_answerable", False):
            continue

        cq = f"{data['scenario']} {data['question']}"
        question_type = qtc(data['answers'])

        success = False
        last_exception = None
        for attempt in range(3):
            try:
                response, prompt_ = custom_query_engine.custom_query(cq, question_type, index_type=args.index, data_FS=None)
                print(response, prompt_ )
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
                "Error": str(last_exception),
            }
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
            continue

        try:
            formatted_response = format_prediction(response.response, question_type)
            if args.eval_method == "bert":
                em, conditional_em, f1, conditional_f1 = scorer_fn(formatted_response, data['answers'], bert_scorer)
            else:
                em, conditional_em, f1, conditional_f1 = scorer_fn(formatted_response, data['answers'])

            scores.append({"em": em, "cem": conditional_em, "f1": f1, "cf1": conditional_f1})

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
                f.write(json.dumps(fallback, ensure_ascii=False) + "\n")

    if skipped_ids:
        with open(skipped_path, "w", encoding="utf-8") as f:
            json.dump(skipped_ids, f, indent=2)
        print(f"[INFO] Skipped {len(skipped_ids)} IDs. Saved to: {skipped_path}")
    elif os.path.exists(skipped_path):
        os.remove(skipped_path)
        print(f"[INFO] All questions processed. Removed old skipped_ids.json")

    # Compute overall average scores
    if scores:
        avg_em = sum(s['em'] for s in scores) / len(scores)
        avg_cem = sum(s['cem'] for s in scores) / len(scores)
        avg_f1 = sum(s['f1'] for s in scores) / len(scores)
        avg_cf1 = sum(s['cf1'] for s in scores) / len(scores)

        overall = {
            "Average EM": avg_em,
            "Average Conditional EM": avg_cem,
            "Average F1": avg_f1,
            "Average Conditional F1": avg_cf1,
            "Total Questions Evaluated": len(scores)
        }

        with open(results_path, "w") as f:
            json.dump(overall, f, indent=2)

        print("\n===== Overall Scores =====")
        for k, v in overall.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print("==========================\n")
