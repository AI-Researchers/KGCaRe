# conditionalqa_eval_bert_score.py

import json
import itertools
import math
import collections
import string
import re
from typing import List, Tuple, Dict, Any

from bert_score import BERTScorer


def load_answers(filename: str) -> Dict[str, Any]:
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    return {d["id"]: d["answers"] for d in data}


def load_and_format_predicted_answers(filename: str) -> List[Dict[str, Any]]:
    with open(filename, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def evaluate_bert_scores(
    pred_file: str,
    ref_file: str,
    scorer: BERTScorer
) -> Dict[str, Dict[str, float]]:
    predictions = load_and_format_predicted_answers(pred_file)
    references = load_answers(ref_file)

    qid2predictions = {d["id"]: d["answers"] for d in predictions}
    qid2references = references

    total_f1, total_cond_f1, total_weighted_score = [], [], []

    for qid, ref_ans in qid2references.items():
        if qid not in qid2predictions:
            continue
        pred_ans = qid2predictions[qid]
        _, _, f1, cond_f1 = compute_meteor_bert_metrics(pred_ans, ref_ans)
        bert_f1, avg_f1, only_cond_f1 = compute_weighted_bert_metrics(pred_ans, ref_ans, scorer)

        if any(ans[1] for ans in ref_ans):  # Conditional
            score = 0.8 * bert_f1 + 0.2 * only_cond_f1
        elif all(ans[0] in ["yes", "no"] for ans in ref_ans):  # Yes/No
            score = f1
        else:  # Span
            score = bert_f1

        total_f1.append(f1)
        total_cond_f1.append(cond_f1)
        total_weighted_score.append(score)

    def avg(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "Scores": {
            "Count": len(total_f1),
            "F1": avg(total_f1),
            "Conditional_F1": avg(total_cond_f1),
            "Weighted_Eval_Score": avg(total_weighted_score),
        }
    }


def compute_meteor_bert_metrics(prediction, reference) -> Tuple[float, float, float, float]:
    if not reference:
        return [float(not prediction)] * 4

    num_answer = len(reference)
    if len(prediction) < num_answer:
        prediction += [("", [])] * (num_answer - len(prediction))

    max_em = max_f1 = max_cond_em = max_cond_f1 = 0.0
    for perm in itertools.permutations(prediction):
        total_em = total_f1 = total_cond_em = total_cond_f1 = 0.0
        for p, r in zip(perm, reference):
            em, cem, f1, cf1 = compute_em_f1(p, r)
            total_em += em
            total_f1 += f1
            total_cond_em += cem
            total_cond_f1 += cf1
        max_em = max(max_em, total_em / num_answer)
        max_f1 = max(max_f1, total_f1 / num_answer)
        max_cond_em = max(max_cond_em, total_cond_em / num_answer)
        max_cond_f1 = max(max_cond_f1, total_cond_f1 / num_answer)

    gamma = math.exp(1.0 - len(prediction) / num_answer)
    return max_em * gamma, max_cond_em * gamma, max_f1 * gamma, max_cond_f1 * gamma


def compute_em_f1(pred, ref):
    cond_f1 = compute_conditions_f1(pred[1], ref[1])
    em = float(normalize_answer(pred[0]) == normalize_answer(ref[0]))
    f1 = compute_answer_f1(normalize_answer(ref[0]), normalize_answer(pred[0]))
    return em, em * cond_f1, f1, f1 * cond_f1


def compute_weighted_bert_metrics(prediction, reference, scorer: BERTScorer):
    if not reference:
        return [float(not prediction)] * 3

    max_bert_f1 = max_avg = max_only_cond = 0.0
    for p in prediction:
        for r in reference:
            b, avg, cond = compute_bert_f1(p, r, scorer)
            max_bert_f1 = max(max_bert_f1, b)
            max_avg = max(max_avg, avg)
            max_only_cond = max(max_only_cond, cond)
    return max_bert_f1, max_avg, max_only_cond


def compute_bert_f1(pred, ref, scorer: BERTScorer):
    b = scorer.score([normalize_answer(pred[0])], [normalize_answer(ref[0])])[2].item()
    if not ref[1]:
        return b, 0, 0
    cond_pred = normalize_answer(" ".join(pred[1]))
    cond_ref = normalize_answer(" ".join(ref[1]))
    cond_f1 = scorer.score([cond_pred], [cond_ref])[2].item()
    return b, (b + cond_f1) / 2.0, cond_f1


def compute_conditions_f1(predicted, true):
    if not true:
        return float(not predicted)
    if not predicted:
        return 0.0
    predicted, true = set(predicted), set(true)
    correct = sum(1 for p in predicted if p in true)
    precision = correct / len(predicted)
    recall = correct / len(true)
    return 2 * precision * recall / (precision + recall) if correct else 0.0


def compute_answer_f1(a_gold, a_pred):
    gold, pred = get_tokens(a_gold), get_tokens(a_pred)
    common = collections.Counter(gold) & collections.Counter(pred)
    num_same = sum(common.values())
    if not gold or not pred:
        return int(gold == pred)
    if num_same == 0:
        return 0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def get_tokens(s): return normalize_answer(s).split() if s else []


def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def run_conditionalqa_bert_evaluation(pred_file: str, ref_file: str, model_name_or_path: str = "microsoft/deberta-large-mnli") -> Dict:
    scorer = BERTScorer(model_name=model_name_or_path)
    result = evaluate_bert_scores(pred_file, ref_file, scorer)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--ref_file", required=True)
    parser.add_argument("--bert_model", default="microsoft/deberta-large-mnli", help="HuggingFace model name for BERTScore")
    args = parser.parse_args()

    result = run_conditionalqa_bert_evaluation(args.pred_file, args.ref_file, args.bert_model)
    print(json.dumps(result, indent=2))
