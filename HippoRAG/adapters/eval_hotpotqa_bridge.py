from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from common import DATA_ROOT, iter_jsonl, write_json

DEFAULT_GOLD_FILE = DATA_ROOT / "stratified_hotpotqa_500sample_with_tag.json"


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def normalize_hotpot_prediction(raw_prediction: str, gold_answer: str) -> str:
    cleaned = raw_prediction.strip()
    normalized_gold = normalize_answer(gold_answer)
    if normalized_gold in ["yes", "no"]:
        tokens = normalize_answer(cleaned).split()
        if "yes" in tokens:
            return "yes"
        if "no" in tokens:
            return "no"
    return cleaned


def load_gold_map(gold_file: Path) -> dict[str, dict[str, Any]]:
    with gold_file.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return {item["_id"]: item for item in data}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge HippoRAG HotpotQA predictions to KGCaRe-compatible evaluation format.")
    parser.add_argument("--input", type=Path, default=None, help="HippoRAG predictions.jsonl")
    parser.add_argument("--gold-file", type=Path, default=DEFAULT_GOLD_FILE)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--eval-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.input is None:
        raise ValueError("--input is required. Point it to an adapter run predictions.jsonl file.")

    input_path = args.input
    output_jsonl = args.output_jsonl or input_path.with_name("output.jsonl")
    eval_json = args.eval_json or input_path.with_name("eval_predictions.json")
    summary_json = args.summary_json or input_path.with_name("results.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {input_path}")
    if not args.gold_file.exists():
        raise FileNotFoundError(f"Missing gold file: {args.gold_file}")

    gold_map = load_gold_map(args.gold_file)

    rows: list[dict[str, Any]] = []
    answer_payload: dict[str, str] = {}

    em_scores: list[float] = []
    f1_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []

    for row in iter_jsonl(input_path):
        qid = row["id"]
        gold_answer = row.get("reference_answer") or gold_map.get(qid, {}).get("answer", "")

        predicted_raw = str(row.get("predicted_answer") or row.get("raw_response") or "")
        normalized_prediction = normalize_hotpot_prediction(predicted_raw, gold_answer)

        em = exact_match_score(normalized_prediction, gold_answer)
        f1, precision, recall = f1_score(normalized_prediction, gold_answer)

        em_scores.append(em)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        answer_payload[qid] = normalized_prediction
        rows.append(
            {
                "_id": qid,
                "question": row.get("question", ""),
                "gold_answer": gold_answer,
                "predicted_answer": normalized_prediction,
                "Score": {
                    "EM": em,
                    "F1": f1,
                    "Precision": precision,
                    "Recall": recall,
                },
                "Raw_Response": row.get("raw_response", ""),
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as file_obj:
        for item in rows:
            file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")

    eval_payload = {"answer": answer_payload, "sp": {}}
    write_json(eval_json, eval_payload)

    summary = {
        "num_predictions": len(rows),
        "averages": {
            "em": mean(em_scores) if em_scores else 0.0,
            "f1": mean(f1_scores) if f1_scores else 0.0,
            "prec": mean(precision_scores) if precision_scores else 0.0,
            "recall": mean(recall_scores) if recall_scores else 0.0,
            "sp_em": 0.0,
            "sp_f1": 0.0,
            "sp_prec": 0.0,
            "sp_recall": 0.0,
            "joint_em": 0.0,
            "joint_f1": 0.0,
            "joint_prec": 0.0,
            "joint_recall": 0.0,
        },
        "paths": {
            "input": str(input_path),
            "output_jsonl": str(output_jsonl),
            "eval_json": str(eval_json),
            "summary_json": str(summary_json),
            "gold_file": str(args.gold_file),
        },
    }
    write_json(summary_json, summary)

    print(f"HotpotQA bridged output: {output_jsonl}")
    print(f"HotpotQA eval payload: {eval_json}")
    print(f"HotpotQA summary: {summary_json}")


if __name__ == "__main__":
    main()
