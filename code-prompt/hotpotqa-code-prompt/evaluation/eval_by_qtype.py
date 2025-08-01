import sys
import ujson as json
import re
import string
from collections import Counter, defaultdict
import pprint

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_metrics(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall

def average_metrics(metrics, count):
    return {k: v / count if count > 0 else 0.0 for k, v in metrics.items()}

def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        predictions = {ex["_id"]: ex["predicted_answer"] for ex in json.load(f)}
    with open(gold_file) as f:
        gold_data = json.load(f)

    grouped_metrics = defaultdict(lambda: {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0})
    counts = defaultdict(int)

    for dp in gold_data:
        qid = dp["_id"]
        qtype = dp.get("qtype", "unknown")
        if qid not in predictions:
            continue
        pred = predictions[qid]
        gold = dp["answer"]
        update_metrics(grouped_metrics[qtype], pred, gold)
        counts[qtype] += 1

    # Joint metrics
    joint_metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    joint_count = 0
    for qtype in grouped_metrics:
        for k in joint_metrics:
            joint_metrics[k] += grouped_metrics[qtype][k]
        joint_count += counts[qtype]

    result = {"joint": average_metrics(joint_metrics, joint_count)}
    for qtype in grouped_metrics:
        result[qtype] = average_metrics(grouped_metrics[qtype], counts[qtype])

    pprint.pprint(result)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python eval_by_qtype.py <prediction_file.json> <gold_file.json>")
        sys.exit(1)
    eval(sys.argv[1], sys.argv[2])
