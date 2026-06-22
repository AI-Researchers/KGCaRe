from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from hybridrag.data import DATA_ROOT, iter_jsonl, write_json


DEFAULT_REF_FILE = DATA_ROOT / "stratified_hotpotqa_500sample_with_tag.json"


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    if normalized_ground_truth in {"yes", "no", "noanswer"} and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_ground_truth.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def normalize_hotpot_prediction(raw_answer: str, gold_answer: str) -> str:
    cleaned = raw_answer.strip()
    normalized_gold = normalize_answer(gold_answer)
    if normalized_gold in {"yes", "no"}:
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


def prediction_from_row(row: dict[str, Any], gold_answer: str) -> str:
    if row.get("normalized_answer") is not None:
        return normalize_hotpot_prediction(str(row["normalized_answer"]), gold_answer)

    structured_answer = row.get("structured_answer")
    if isinstance(structured_answer, dict) and structured_answer.get("answer") is not None:
        return normalize_hotpot_prediction(str(structured_answer["answer"]), gold_answer)

    return normalize_hotpot_prediction(str(row.get("raw_response", "")), gold_answer)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and evaluate HybridRAG HotPotQA predictions with EM/F1.")
    parser.add_argument("--input", type=Path, required=True, help="HybridRAG HotPotQA predictions JSONL.")
    parser.add_argument("--gold-file", type=Path, default=DEFAULT_REF_FILE)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--eval-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = args.input
    output_jsonl = args.output_jsonl or input_path.with_name("normalized_output.jsonl")
    eval_json = args.eval_json or input_path.with_name("eval_predictions.json")
    summary_json = args.summary_json or input_path.with_name("results.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {input_path}")
    if not args.gold_file.exists():
        raise FileNotFoundError(f"Missing gold file: {args.gold_file}")

    gold_map = load_gold_map(args.gold_file)
    detailed_rows: list[dict[str, Any]] = []
    answer_payload: dict[str, str] = {}
    em_scores: list[float] = []
    f1_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in iter_jsonl(input_path):
        qid = row["id"]
        gold_item = gold_map.get(qid, {})
        gold_answer = row.get("reference_answer")
        if gold_answer is None:
            gold_answer = gold_item.get("answer", "")
        question_type = row.get("question_type") or gold_item.get("qtype") or gold_item.get("type") or ""
        normalized_prediction = prediction_from_row(row, str(gold_answer))

        em = exact_match_score(normalized_prediction, str(gold_answer))
        f1, precision, recall = f1_score(normalized_prediction, str(gold_answer))
        em_scores.append(em)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        by_type[str(question_type)]["EM"].append(em)
        by_type[str(question_type)]["F1"].append(f1)
        by_type[str(question_type)]["Precision"].append(precision)
        by_type[str(question_type)]["Recall"].append(recall)

        answer_payload[qid] = normalized_prediction
        detailed_rows.append(
            {
                "_id": qid,
                "question": row.get("question", ""),
                "question_type": question_type,
                "gold_answer": gold_answer,
                "predicted_answer": normalized_prediction,
                "raw_response": row.get("raw_response", ""),
                "parse_mode": row.get("parse_mode", ""),
                "Score": {
                    "EM": em,
                    "F1": f1,
                    "Precision": precision,
                    "Recall": recall,
                },
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as file_obj:
        for row in detailed_rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_json(eval_json, {"answer": answer_payload, "sp": {}})

    type_breakdown = {
        question_type: {
            metric: mean(values) if values else 0.0
            for metric, values in metric_map.items()
        }
        for question_type, metric_map in by_type.items()
    }
    summary = {
        "num_predictions": len(detailed_rows),
        "num_gold": len(gold_map),
        "missing_predictions": max(0, len(gold_map) - len(detailed_rows)),
        "averages": {
            "EM": mean(em_scores) if em_scores else 0.0,
            "F1": mean(f1_scores) if f1_scores else 0.0,
            "Precision": mean(precision_scores) if precision_scores else 0.0,
            "Recall": mean(recall_scores) if recall_scores else 0.0,
        },
        "type_breakdown": type_breakdown,
        "paths": {
            "input": str(input_path),
            "normalized_output": str(output_jsonl),
            "eval_json": str(eval_json),
            "summary": str(summary_json),
            "gold_file": str(args.gold_file),
        },
    }
    write_json(summary_json, summary)

    print(f"HotPotQA normalized output: {output_jsonl}")
    print(f"HotPotQA eval payload: {eval_json}")
    print(f"HotPotQA summary: {summary_json}")


if __name__ == "__main__":
    main()
