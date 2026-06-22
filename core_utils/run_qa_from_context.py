#!/usr/bin/env python3
import argparse, json, time, traceback, subprocess
from pathlib import Path
from tqdm import tqdm
from typing import List
import faiss
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core_utils")))


from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.postprocessor import LLMRerank
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import KGTableRetriever, VectorIndexRetriever
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

SYSTEM_PROMPT = (
    "You are an expert multi‑hop reasoning assistant. "
    "Your task: take help from the provided context and knowledge graph triples, "
    "and provide the **final concise answer only**—no explanations, no filler words."
    "\n- For yes/no questions, respond with exactly **Yes** or **No**."
    "\n- For span or factoid answers, respond with **only the answer**."
    "\n\nUnderstand how to answer from EXAMPLES:\n"
    "Q: Which city is the birth place of the author of The Shining?\n"
    "A: Portland, Maine\n\n"
    "Q: Which country has a smaller population, the one that uses the rial or the one that uses the rupee?\n"
    "A: Oman\n\n"
    "Q: Was the person who played Frodo in The Lord of the Rings born before or after the person who played Harry Potter in the movies?\n"
    "A: Before\n\n"
    "Q: Is the number of studio albums released by Nirvana more or less than the number of studio albums released by Pink Floyd?\n"
    "A: Less\n\n"
    "Q: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?\n"
    "A: No\n\n"
    "Q: Were Scott Derrickson and Ed Wood of the same nationality?\n"
    "A: Yes\n\n"
)

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Input JSONL with context.")
    p.add_argument("--output", type=Path, required=True, help="Output prediction JSONL file.")
    p.add_argument("--gold_data", type=Path, required=True, help="Output JSONL file.")
    p.add_argument("--model", type=str, required=True, help="LLM model to use (e.g.,'gpt-4o' 'gpt-3.5-turbo','mistralai/Mixtral-8x7B-Instruct-v0.1','mistralai/Mistral-7B-Instruct-v0.3')")
    p.add_argument("--max-new-tokens", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--framework", type=str, default='openai', choices=['openai', 'openai-like'])
    p.add_argument("--context", type=str, default='hybrid', choices=['hybrid', 'vector', 'kg', 'no-context'])
    return p

# ----------------------------------------------------------------------
# Retrieval loaders
# ----------------------------------------------------------------------
def load_vector_retriever(storage_dir: Path, embed_model, similarity_top_k: int):
    vector_store = FaissVectorStore.from_persist_dir(str(storage_dir))
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(storage_dir))
    vector_index = load_index_from_storage(storage_context=storage_ctx,index_id="vector_index_dev")
    return VectorIndexRetriever(embed_model=embed_model, index=vector_index, similarity_top_k=similarity_top_k)

def convert_jsonl_to_eval_json(pred_jsonl: Path, eval_json: Path):
    answer_dict = {}
    with pred_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                answer_dict[obj["_id"]] = obj["predicted_answer"]
    final = {"answer": answer_dict, "sp": {}}
    with eval_json.open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

def evaluate(pred_jsonl: Path, gold_json: Path, log_file: Path):
    eval_json = pred_jsonl.with_suffix(".eval.json")
    convert_jsonl_to_eval_json(pred_jsonl, eval_json)
    cmd = ["python", "hotpot_evaluate_v1.py", str(eval_json), str(gold_json)]
    print("🔎 Running evaluation:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout.strip() or proc.stderr.strip()
    print(out)
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n## Eval for {pred_jsonl.name}\n{out}\n")

def main(args):
    if args.framework == 'openai-like':
            # Setup LLM and Embedding Model
        llm = OpenAILike(
            model=args.model,                                     #"mistralai/Mixtral-8x7B-Instruct-v0.1","mistralai/Mistral-7B-Instruct-v0.3"
            api_base=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "not_needed"),
            is_chat_model=True,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    elif args.framework == 'openai':
        llm = OpenAI(
            model=args.model,
        )
    
    Settings.llm = llm
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    Settings.embed_model = embed_model
    
    reranker = LLMRerank(llm=llm, choice_batch_size=10, top_n=3)
    
    vector_retriever = None
    kg_retriever = None
    if args.context in {"vector", "hybrid"}:
        vector_retriever = load_vector_retriever("./storage/dev_wiki_articles_500_neo4j_faiss_kg_v3", embed_model, 2)


    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()

    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f_out:
            for line in f_out:
                try:
                    completed.add(json.loads(line)["_id"])
                except Exception:
                    continue

    with args.input.open("r", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin if line.strip()]

    system_msg = ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)

    MAX_RETRIES = 3
    BACKOFFS = (2, 5, 15)

    with args.output.open("a", encoding="utf-8") as fout:
        for item in tqdm(data, desc="Answering"):
            qid = item["_id"]
            if qid in completed:
                continue

            vector_context = item.get("vector_context", "")
            kg_context = item.get("kg_context", "")
            question = item["question"]
            gold_answer = item.get("answer")

            user_msg_content = ""
            if args.context == 'hybrid':
                context_chunks: List[str] = []
                retrieved_nodes = vector_retriever.retrieve(question)
                # retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=question)
                context_chunks.extend(node.get_content() for node in retrieved_nodes)
                vector_context = "\n\n".join(context_chunks)
                user_msg_content += f"Context:\n Knowledge Graph Triples:\n{kg_context}\n\n Text : \n {vector_context} \n\n"
            if args.context == 'vector':
                context_chunks: List[str] = []
                retrieved_nodes = vector_retriever.retrieve(question)
                # retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=question)
                context_chunks.extend(node.get_content() for node in retrieved_nodes)
                vector_context = "\n\n".join(context_chunks)
                user_msg_content += f"Context:\n{vector_context}\n\n"
            elif args.context == 'kg':
                user_msg_content += f"Context:\n Knowledge Graph Triples:\n{kg_context}\n\n"
            elif args.context == 'no-context':
                user_msg_content += "Context:\n No context provided. \n Answer the question with your own knowledge !\n\n"
            user_msg_content += f"Q: {question}\nA:"
            # print(user_msg_content)
            msgs = [system_msg, ChatMessage(role=MessageRole.USER, content=user_msg_content)]
            answer = ""

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    if args.framework == 'openai':
                        resp = llm.chat(msgs)
                    elif args.framework == 'openai-like':
                        resp = llm.chat(msgs,stop=["\n<"])
                    answer = resp.message.content.strip()
                    break
                except Exception as exc:
                    if attempt == MAX_RETRIES:
                        traceback.print_exc()
                        answer = f"<<ERROR after {MAX_RETRIES} retries: {exc}>>"
                    else:
                        time.sleep(BACKOFFS[min(attempt - 1, len(BACKOFFS) - 1)])

            fout.write(json.dumps({
                "_id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": answer,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            completed.add(qid)

    # Convert to array JSON
    array_out = args.output.with_suffix(".json")
    print(f"📦 Converting to array {array_out} ...")
    with args.output.open("r", encoding="utf-8") as fin:
        preds = [json.loads(line) for line in fin if line.strip()]
    with array_out.open("w", encoding="utf-8") as fout:
        json.dump(preds, fout, indent=2, ensure_ascii=False)
    print(f"✅ JSON array saved: {array_out}")

    # Run evaluation
    evaluate(args.output, args.gold_data, Path("hotpot_eval.log"))

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)
