#!/usr/bin/env python3
"""
Extract topic entities from each HotpotQA question via LLM.
Restart‑safe: will skip completed IDs if re-run.

Output:
-------
* JSONL file with {"_id": str, "question": str, "entities": [list of str]}
* Final .json array file.

Usage:
------
python hotpotqa_entity_extraction.py \
  --input hotpot_dev_distractor.json \
  --output entities.jsonl \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1
"""
from __future__ import annotations
import argparse, json, os, time, traceback
from pathlib import Path
from typing import List
from tqdm import tqdm
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole

# ============================= Prompt ======================================
SYSTEM_PROMPT = (
    "You are an expert in knowledge graph reasoning and entity extraction.\n"
    "TASK: Given a natural language question, extract up to 15 relavent keywords from the question. "
    "Focus on extracting the keywords that we can use to best lookup answers to the question. Avoid stopwords.\n"
    "IMPORTANT:\n"
    "- Output as a JSON list of entity strings.\n"
    "- Include both EXPLICIT entities (directly mentioned in the question) and IMPLICIT entities (concepts implied by the question, such as attributes, relations, or needed comparison terms).\n"
    "- Be precise. Do not add explanations, only output a valid JSON array.\n\n"
    "EXAMPLES:\n\n"
    "Q: Which city is the birth place of the author of The Shining?\n"
    "Entities: [\"The Shining\", \"author\", \"birth place\", \"city\"]\n\n"
    "Q: Which country has a smaller population, the one that uses the rial or the one that uses the rupee?\n"
    "Entities: [\"country\", \"population\", \"rial\", \"rupee\", \"currency\"]\n\n"
    "Q: Was the person who played Frodo in The Lord of the Rings born before or after the person who played Harry Potter in the movies?\n"
    "Entities: [\"Frodo\", \"The Lord of the Rings\", \"Harry Potter\", \"movies\", \"birth date\", \"actor\"]\n\n"
    "Q: Is the number of studio albums released by Nirvana more or less than the number of studio albums released by Pink Floyd?\n"
    "Entities: [\"Nirvana\", \"Pink Floyd\", \"studio albums\", \"number of albums\", \"music artist\"]\n\n"
    "Q: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?\n"
    "Entities: [\"Laleli Mosque\", \"Esma Sultan Mansion\", \"neighborhood\", \"location\"]\n\n"
    "Q: Were Scott Derrickson and Ed Wood of the same nationality?\n"
    "Entities: [\"Scott Derrickson\", \"Ed Wood\", \"nationality\", \"person\"]\n\n"
)


# ============================= CLI args =====================================
def build_arg_parser():
    p = argparse.ArgumentParser(description="Extract topic entities from HotpotQA questions.")
    p.add_argument("--input", type=Path, required=True, help="Path to HotpotQA JSON.")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL file.")
    p.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model name.")
    p.add_argument("--api-base", type=str, default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", "not_needed"))
    p.add_argument("--max-new-tokens", type=int, default=200, help="Max new tokens.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    return p

# ============================= Main =========================================
def main():
    args = build_arg_parser().parse_args()

    llm = OpenAILike(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        is_chat_model=True,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # Load input
    data = json.loads(args.input.read_text(encoding="utf-8"))

    # Track completed
    completed_ids = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["_id"])
                except Exception:
                    continue
        print(f"▶️  {len(completed_ids)} already completed, {len(data)-len(completed_ids)} remaining.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as fout:
        for item in tqdm(data, desc="Extracting", unit="q"):
            qid = item["_id"]
            if qid in completed_ids:
                continue

            question = item["question"]
            ans = item.get("answer", "")
            gold_answer = str(ans).strip().lower()
            answer_type = "yes/no" if gold_answer in {"yes", "no"} else "span"
            
            user_msg = f"Q: {question}\n Entities:"
            msgs = [
                ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER, content=user_msg),
            ]

            # Call with retries
            answer = ""
            MAX_RETRIES = 3
            BACKOFF = [2, 5, 15]
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = llm.chat(msgs,stop=["\n<"])
                    answer = resp.message.content.strip()
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        traceback.print_exc()
                        answer = "[]"
                    else:
                        wait = BACKOFF[min(attempt-1, len(BACKOFF)-1)]
                        print(f"⚠️ Error: {e} retrying in {wait}s...")
                        time.sleep(wait)

            # Parse JSON output
            entities: List[str] = []
            try:
                parsed = json.loads(answer)
                if isinstance(parsed, list):
                    entities = [str(x) for x in parsed]
            except Exception:
                # try to salvage
                try:
                    start = answer.find("[")
                    end = answer.rfind("]")
                    if start != -1 and end != -1:
                        partial = answer[start:end+1]
                        entities = json.loads(partial)
                except Exception:
                    entities = []

            fout.write(json.dumps({"_id": qid, "question": question, "answer":ans,"qtype": answer_type,"entities": entities}, ensure_ascii=False) + "\n")
            fout.flush()
            completed_ids.add(qid)

    # Convert to final JSON
    out_json = args.output.with_suffix(".json")
    preds = []
    with args.output.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                preds.append(json.loads(line))
            except Exception as e:
                print(f"⚠️ Skipping invalid line: {e}")
    out_json.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
    print(f"✅ All done. JSON saved to {out_json}")

if __name__ == "__main__":
    main()
