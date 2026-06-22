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

from core_utils.conditionalqa_eval import compute_metrics, run_conditionalqa_evaluation  # noqa: E402
from core_utils.format_utils import format_prediction  # noqa: E402

DEFAULT_REF_FILE = DATA_ROOT / "dev.json"


def load_reference_map(ref_file: Path) -> dict[str, list[list[Any]]]:
    with ref_file.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return {item["id"]: item["answers"] for item in data}


def question_type_from_reference(reference_answers: list[list[Any]]) -> str:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge HippoRAG ConditionalQA predictions to KGCaRe-compatible evaluation format.")
    parser.add_argument("--input", type=Path, default=None, help="HippoRAG predictions.jsonl")
    parser.add_argument("--ref-file", type=Path, default=DEFAULT_REF_FILE)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--eval-pred-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    default_input = dataset_paths("ConditionalQA")

    input_path = args.input
    if input_path is None:
        raise ValueError("--input is required. Point it to an adapter run predictions.jsonl file.")

    output_jsonl = args.output_jsonl or input_path.with_name("output.jsonl")
    eval_pred_jsonl = args.eval_pred_jsonl or input_path.with_name("eval_predictions.jsonl")
    summary_json = args.summary_json or input_path.with_name("results.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {input_path}")
    if not args.ref_file.exists():
        raise FileNotFoundError(f"Missing reference file: {args.ref_file}")

    ref_map = load_reference_map(args.ref_file)

    output_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    em_scores: list[float] = []
    cem_scores: list[float] = []
    f1_scores: list[float] = []
    cf1_scores: list[float] = []

    for row in iter_jsonl(input_path):
        qid = row["id"]
        reference_answers = row.get("reference_answers") or ref_map.get(qid, [])
        question_type = row.get("question_type") or question_type_from_reference(reference_answers)

        raw_response = str(row.get("raw_response", "")).strip()
        if not raw_response:
            raw_response = str(row.get("predicted_answer", "")).strip()

        normalized_answer = format_prediction(raw_response, question_type, dataset="conditionalqa")
        if reference_answers and len(normalized_answer) > len(reference_answers):
            normalized_answer = normalized_answer[: len(reference_answers)]

        em, cem, f1, cf1 = compute_metrics(normalized_answer, reference_answers)
        em_scores.append(em)
        cem_scores.append(cem)
        f1_scores.append(f1)
        cf1_scores.append(cf1)

        output_rows.append(
            {
                "id": qid,
                "Prompt": row.get("model_query", ""),
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
                "Raw_Response": row.get("raw_response", ""),
            }
        )
        eval_rows.append({"id": qid, "answer": normalized_answer})

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as file_obj:
        for item in output_rows:
            file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")

    with eval_pred_jsonl.open("w", encoding="utf-8") as file_obj:
        for item in eval_rows:
            file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")

    type_breakdown, type_map, filtered_count = run_conditionalqa_evaluation(
        pred_file=str(eval_pred_jsonl),
        ref_file=str(args.ref_file),
        question_type="all_types",
    )

    summary = {
        "num_predictions": len(output_rows),
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
            "output_jsonl": str(output_jsonl),
            "eval_pred_jsonl": str(eval_pred_jsonl),
            "summary_json": str(summary_json),
        },
    }
    write_json(summary_json, summary)

    print(f"ConditionalQA bridged output: {output_jsonl}")
    print(f"ConditionalQA eval predictions: {eval_pred_jsonl}")
    print(f"ConditionalQA summary: {summary_json}")


if __name__ == "__main__":
    main()
