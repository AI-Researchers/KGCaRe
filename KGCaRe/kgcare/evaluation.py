from __future__ import annotations

import collections
import itertools
import json
import math
import re
import string
from pathlib import Path
from typing import Any, Dict


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split() if s else []


def _answer_f1(gold: str, pred: str) -> float:
    gold_toks = _tokens(gold)
    pred_toks = _tokens(pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def _conditions_f1(predicted: list[str], true: list[str]) -> float:
    if not true:
        return float(not predicted)
    if not predicted:
        return 0.0
    true = list(set(true))
    predicted = list(set(predicted))
    correct = sum(int(c in true) for c in predicted)
    if correct == 0:
        return 0.0
    precision = correct / len(predicted)
    recall = correct / len(true)
    return 2.0 / (1.0 / precision + 1.0 / recall)


def _conditionalqa_pair_metrics(pred_answer: list[Any], ref_answer: list[Any]) -> tuple[float, float, float, float]:
    conditions_f1 = _conditions_f1(pred_answer[1], ref_answer[1])
    pred_text = normalize_answer(pred_answer[0])
    ref_text = normalize_answer(ref_answer[0])
    em = float(pred_text == ref_text)
    f1 = _answer_f1(ref_text, pred_text)
    return em, em * conditions_f1, f1, f1 * conditions_f1


def conditionalqa_compute_metrics(prediction: list[list[Any]], reference: list[list[Any]]) -> tuple[float, float, float, float]:
    if not reference:
        return tuple([float(not prediction)] * 4)  # type: ignore[return-value]
    prediction = list(prediction)
    num_answer = len(reference)
    if len(prediction) < num_answer:
        prediction.extend([["", []]] * (num_answer - len(prediction)))
    max_em = max_cem = max_f1 = max_cf1 = 0.0
    for ordered_prediction in itertools.permutations(prediction):
        total_em = total_cem = total_f1 = total_cf1 = 0.0
        for pred_answer, ref_answer in zip(ordered_prediction, reference):
            em, cem, f1, cf1 = _conditionalqa_pair_metrics(pred_answer, ref_answer)
            total_em += em
            total_cem += cem
            total_f1 += f1
            total_cf1 += cf1
        max_em = max(max_em, total_em / num_answer)
        max_cem = max(max_cem, total_cem / num_answer)
        max_f1 = max(max_f1, total_f1 / num_answer)
        max_cf1 = max(max_cf1, total_cf1 / num_answer)
    gamma = math.exp(1.0 - len(prediction) / num_answer)
    return max_em * gamma, max_cem * gamma, max_f1 * gamma, max_cf1 * gamma


def evaluate_conditionalqa(pred_jsonl: Path, ref_file: Path) -> dict[str, Any]:
    ref_data = [item for item in json.loads(ref_file.read_text(encoding="utf-8")) if not item.get("not_answerable", False)]
    predictions: dict[str, Any] = {}
    for line in pred_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        predictions[item["id"]] = item["answer"]

    buckets = {"total": [], "yesno": [], "extractive": [], "conditional": []}
    for idx, item in enumerate(ref_data):
        answers = item["answers"]
        if any(ans[0] in ["yes", "no"] for ans in answers):
            buckets["yesno"].append(idx)
        else:
            buckets["extractive"].append(idx)
        if any(ans[1] for ans in answers):
            buckets["conditional"].append(idx)
        buckets["total"].append(idx)

    scores = []
    for item in ref_data:
        pred = predictions.get(item["id"], [])
        scores.append(conditionalqa_compute_metrics(pred, item["answers"]))

    def summarize(indices: list[int]) -> dict[str, float]:
        if not indices:
            return {"EM": 0.0, "EM_with_conditions": 0.0, "F1": 0.0, "F1_with_conditions": 0.0}
        return {
            "EM": sum(scores[i][0] for i in indices) / len(indices),
            "EM_with_conditions": sum(scores[i][1] for i in indices) / len(indices),
            "F1": sum(scores[i][2] for i in indices) / len(indices),
            "F1_with_conditions": sum(scores[i][3] for i in indices) / len(indices),
        }

    return {name: summarize(indices) for name, indices in buckets.items()} | {"count": len(ref_data)}


def _hotpot_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    zero = (0.0, 0.0, 0.0)
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def evaluate_hotpotqa(pred_jsonl: Path, ref_file: Path) -> dict[str, float]:
    predictions: dict[str, str] = {}
    for line in pred_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        predictions[item["_id"]] = str(item["predicted_answer"])
    gold = json.loads(ref_file.read_text(encoding="utf-8"))
    metrics = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    for item in gold:
        pred = predictions.get(item["_id"], "")
        answer = item["answer"]
        metrics["em"] += float(normalize_answer(pred) == normalize_answer(answer))
        f1, prec, recall = _hotpot_f1(pred, answer)
        metrics["f1"] += f1
        metrics["prec"] += prec
        metrics["recall"] += recall
    n = len(gold) or 1
    return {key: value / n for key, value in metrics.items()} | {"count": len(gold)}
