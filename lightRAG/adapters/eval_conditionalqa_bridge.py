from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from common import DATA_ROOT, dataset_paths, iter_jsonl, write_json

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(WORKSPACE_ROOT))

from core_utils.format_utils import format_prediction  # noqa: E402
from core_utils.conditionalqa_eval import compute_metrics, run_conditionalqa_evaluation  # noqa: E402


DEFAULT_REF_FILE = DATA_ROOT / "dev.json"


def classify_question_type_from_reference(reference_answers: list[list[Any]]) -> str:
    if not reference_answers:
        return "unanswerable"
    yes_no = any(ans[0] in ["yes", "no"] for ans in reference_answers)
    has_conditions = any(ans[1] for ans in reference_answers)
    if yes_no and has_conditions:
        return "yes/no_conditional"
    if yes_no:
        return "yes/no"
    if has_conditions:
        return "span_conditional"
    return "span"



def load_reference_map(ref_file: Path) -> dict[str, list[list[Any]]]:
    with ref_file.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return {item["id"]: item["answers"] for item in data}


def normalize_from_structured_answer(structured_answer: dict[str, Any] | None) -> list[list[Any]] | None:
    if not structured_answer:
        return None
    answers = structured_answer.get("answers") or []
    conditions = structured_answer.get("conditions") or []
    if not answers:
        return None

    while len(conditions) < len(answers):
        conditions.append([])

    normalized: list[list[Any]] = []
    for index, answer in enumerate(answers):
        answer_text = str(answer).strip()
        answer_type = structured_answer.get("answer_type", "span")
        if answer_type == "yes_no":
            lowered = answer_text.lower()
            if lowered.startswith("yes"):
                answer_text = "yes"
            elif lowered.startswith("no"):
                answer_text = "no"
        normalized.append([answer_text, list(map(str, conditions[index]))])
    return normalized



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and evaluate LightRAG predictions with ConditionalQA metrics.")
    parser.add_argument("--input", type=Path, default=None, help="Raw LightRAG predictions JSONL.")
    parser.add_argument("--ref-file", type=Path, default=DEFAULT_REF_FILE)
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Detailed normalized output JSONL.")
    parser.add_argument("--eval-pred-jsonl", type=Path, default=None, help="JSONL file in ConditionalQA eval schema (id + answer).")
    parser.add_argument("--summary-json", type=Path, default=None, help="Aggregated summary JSON output.")
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    paths = dataset_paths("ConditionalQA")

    input_path = args.input or paths.predictions_path
    output_jsonl = args.output_jsonl or input_path.with_name("normalized_output.jsonl")
    eval_pred_jsonl = args.eval_pred_jsonl or input_path.with_name("eval_predictions.jsonl")
    summary_json = args.summary_json or input_path.with_name("results.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input predictions: {input_path}")
    if not args.ref_file.exists():
        raise FileNotFoundError(f"Missing reference file: {args.ref_file}")

    ref_map = load_reference_map(args.ref_file)
    detailed_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    em_scores: list[float] = []
    cem_scores: list[float] = []
    f1_scores: list[float] = []
    cf1_scores: list[float] = []

    for row in iter_jsonl(input_path):
        qid = row["id"]
        raw_response = row.get("raw_response", "")
        reference_answers = row.get("reference_answers") or ref_map.get(qid, [])
        question_type = row.get("question_type") or classify_question_type_from_reference(reference_answers)

        normalized_answer = normalize_from_structured_answer(row.get("structured_answer"))
        if normalized_answer is None:
            normalized_answer = format_prediction(raw_response, question_type, dataset="conditionalqa")
        if reference_answers and len(normalized_answer) > len(reference_answers):
            normalized_answer = normalized_answer[: len(reference_answers)]
        em, cem, f1, cf1 = compute_metrics(normalized_answer, reference_answers)

        em_scores.append(em)
        cem_scores.append(cem)
        f1_scores.append(f1)
        cf1_scores.append(cf1)

        detailed_rows.append(
            {
                "id": qid,
                "Question": row.get("question", ""),
                "Question_Type": question_type,
                "answers": normalized_answer,
                "Actual_Answer": reference_answers,
                "Score": {
                    "EM": em,
                    "Conditional_EM": cem,
                    "F1": f1,
                    "Conditional_F1": cf1,
                },
                "Raw_Response": raw_response,
            }
        )
        eval_rows.append({"id": qid, "answer": normalized_answer})

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    eval_pred_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as file_obj:
        for row in detailed_rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")

    with eval_pred_jsonl.open("w", encoding="utf-8") as file_obj:
        for row in eval_rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")

    type_breakdown, type_map, filtered_count = run_conditionalqa_evaluation(
        pred_file=str(eval_pred_jsonl),
        ref_file=str(args.ref_file),
        question_type="all_types",
    )

    summary = {
        "num_predictions": len(detailed_rows),
        "filtered_count": filtered_count,
        "averages": {
            "Average EM": mean(em_scores) if em_scores else 0.0,
            "Average Conditional EM": mean(cem_scores) if cem_scores else 0.0,
            "Average F1": mean(f1_scores) if f1_scores else 0.0,
            "Average Conditional F1": mean(cf1_scores) if cf1_scores else 0.0,
        },
        "type_breakdown": type_breakdown,
        "type_map_size": len(type_map),
        "paths": {
            "input": str(input_path),
            "normalized_output": str(output_jsonl),
            "eval_predictions": str(eval_pred_jsonl),
            "summary": str(summary_json),
        },
    }
    write_json(summary_json, summary)

    print(f"ConditionalQA normalized output: {output_jsonl}")
    print(f"ConditionalQA eval predictions: {eval_pred_jsonl}")
    print(f"ConditionalQA summary: {summary_json}")


if __name__ == "__main__":
    main()
