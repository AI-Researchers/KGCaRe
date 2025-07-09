#!/usr/bin/env python3

"""
Official evaluation script of ConditionalQA.

To run this script (python3):
  python evaluate.py --pred_file=PATH_TO_YOUR_FILE --ref_file=PATH_TO_REF

"""

import json
import itertools
import math
import collections
import string
import re
import argparse
import pandas as pd
from bert_score import BERTScorer


def evaluate(results_filename, reference_filename, excel_filename):
    """Compute evaluation metrics."""
    # predictions = load_and_format_predicted_answers(results_filename)
    predictions = load_and_format_predicted_answers_v2(results_filename)
    qid2predictions = {d["id"]: d["answers"] for d in predictions}
    qid2references = load_answers(reference_filename)
    # qid2references, qid2question, qid2qtype = load_ref_answers_from_predicted_file(results_filename)
    # qid2references = {d["id"]: d["answers"] for d in qid2references}
    # print("pred : ", qid2predictions)
    # print("\nref : ", qid2references)

    # Load the BERT model once and reuse it
    bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli",lang="en")
    
    total_em, total_f1, total_conditional_f1 = list(), list(), list()
    total_bert_f1, total_avg_ans_conditions_bert_f1, total_only_conditional_bert_f1 = list(), list(), list()
    total_weighted_score = list()

    yesno_questions = list()
    extractive_questions = list()
    conditional_questions = list()

    only_yesno_f1, yesno_cond_f1, only_span_f1, span_cond_f1 = list(),list(),list(),list()
    all_yesno_f1, all_span_f1 = list(),list()
    all_cond_score = list()
    all_ques_score = list()

    i = 0
    output_list = []
    yesno_cond_count = 0
    yesno_only_count = 0
    span_cond_count = 0
    span_only_count = 0
    not_answerable =0
    bert_f1, avg_ans_conditions_bert_f1, only_conditional_bert_f1 = 0,0,0
    for _, qid in enumerate(qid2references.keys()):
        
        # if qid in ["dev-15","dev-14"]:
        #     print(qid, qid2references[qid], len(qid2references[qid]))
        #     print(qid2predictions[qid])

        if len(qid2references[qid]) == 0:
            continue
        else:

            if qid not in qid2predictions:
                print("qid not in pred!!!")
                em, conditional_em, f1, conditional_f1 = 0.0, 0.0, 0.0, 0.0
            else:
                em, conditional_em, f1, conditional_f1 = compute_metrics(
                    qid2predictions[qid], qid2references[qid]
                )

                bert_f1, avg_ans_conditions_bert_f1, only_conditional_bert_f1 = compute_weighted_bert_metrics_V2(
                qid2predictions[qid], qid2references[qid], bert_scorer
                )

                row = {
                    "Id": qid,
                    # "Question": qid2question[qid],
                    "Reference": qid2references[qid],
                    "Predicted": qid2predictions[qid],
                    # "Question_type": qid2qtype[qid],
                    "Only_Ans_F1":f1,
                    "Ans_Conditions_F1":conditional_f1, 
                    "Only_Ans_F1_BERTScore": bert_f1,
                    "Ans_Conditions_F1_BERTScore": avg_ans_conditions_bert_f1,
                    "Only_Conditions_F1_BERTScore": only_conditional_bert_f1
                }
                
                # total_em.append(em)
                total_f1.append(f1)
                total_conditional_f1.append(conditional_f1)
                total_bert_f1.append(bert_f1)
                total_avg_ans_conditions_bert_f1.append(avg_ans_conditions_bert_f1)
                total_only_conditional_bert_f1.append(only_conditional_bert_f1)
                
                # print("---conditional_f1 = ", conditional_f1)
                if not qid2references[qid]:
                    pass
                elif any(ans[0] in ["yes", "no"] for ans in qid2references[qid]):
                    yesno_questions.append(i)
                else:
                    extractive_questions.append(i)

                if any(ans[1] for ans in qid2references[qid]):
                    conditional_questions.append(i)
                
                if any(ans[1] for ans in qid2references[qid]): # conditional question
                    
                    if any(ans[0] in ["yes", "no"] for ans in qid2references[qid]): #yes no condition
                        yesno_cond_count += 1
                        new_eval_score = (0.8 * f1) + (0.2 * only_conditional_bert_f1)
                        yesno_cond_f1.append(new_eval_score)
                        all_yesno_f1.append(new_eval_score)
                        
                    else:
                        span_cond_count += 1
                        new_eval_score = (0.8 * bert_f1) + (0.2 * only_conditional_bert_f1)
                        span_cond_f1.append(new_eval_score)
                        all_span_f1.append(new_eval_score)
                        
                    all_cond_score.append(new_eval_score)
                    all_ques_score.append(new_eval_score)
                
                elif any(ans[0] in ["yes", "no"] for ans in qid2references[qid]): #yesno but no condition
                    yesno_only_count += 1
                    new_eval_score = f1
                    only_yesno_f1.append(new_eval_score)
                    all_yesno_f1.append(new_eval_score)
                    all_ques_score.append(new_eval_score)
                
                if len(qid2references[qid]) == 0:
                    not_answerable += 1
                elif not any(ans[0] in ["yes", "no"] for ans in qid2references[qid]) and not any(ans[1] for ans in qid2references[qid]):
                    span_only_count += 1
                    new_eval_score = bert_f1
                    only_span_f1.append(new_eval_score)
                    all_span_f1.append(new_eval_score)
                    all_ques_score.append(new_eval_score)

                total_weighted_score.append(new_eval_score)
                row['Weighted_Score'] = new_eval_score
                
                output_list.append(row)

                i += 1
    #             continue
    # else:
    #     continue

    def update_metrics(questions, prefix=""):
        
        return {
            prefix + "Len of Questions = ": len(questions),
            prefix + "Only_Ans_F1": sum(total_f1[i] for i in questions) / len(questions)
            if len(questions) > 0
            else 0.0,
            prefix
            + "Ans_conditions_F1": sum(total_conditional_f1[i] for i in questions)
            / len(questions)
            if len(questions) > 0
            else 0.0,
            prefix + "Only_Ans_F1_BERTScore": sum(total_bert_f1[i] for i in questions) / len(questions)
            if len(questions) > 0
            else 0.0,
            prefix
            + "Ans_Conditions_F1_BERTScore": sum(total_avg_ans_conditions_bert_f1[i] for i in questions)
            / len(questions)
            if len(questions) > 0
            else 0.0,
            prefix
            + "Only_Conditions_BERT_F1": sum(total_only_conditional_bert_f1[i] for i in questions)
            / len(questions)
            if len(questions) > 0
            else 0.0,
            prefix
            + "Weighted_Score": sum(total_weighted_score[i] for i in questions)
            / len(questions)
            if len(questions) > 0
            else 0.0,
        }

    def update_metrics_new(answers, prefix=""):
    
        return {
            prefix + "Len of answers = ": len(answers),
            
            prefix + "New Score": sum(answers) / len(answers)
            if len(answers) > 0
            else 0.0,
        }

    # print(total_conditional_f1)

    # output_df = pd.DataFrame(output_list)
    # output_df.to_excel(excel_filename, index=False)

    
    
    return {
        "total": update_metrics(range(len(total_f1))),
        "total New": update_metrics_new(all_ques_score),
        
        "yesno": update_metrics(yesno_questions),
        "ALL yesno NEW": update_metrics_new(all_yesno_f1),
        "yesno ONLY NEW": update_metrics_new(only_yesno_f1),
        "yesno COND NEW": update_metrics_new(yesno_cond_f1),
        
        "extractive": update_metrics(extractive_questions),
        "ALL span NEW": update_metrics_new(all_span_f1),
        "span ONLY NEW": update_metrics_new(only_span_f1),
        "span COND NEW": update_metrics_new(span_cond_f1),
        
        "conditional": update_metrics(conditional_questions),
        "conditional NEW": update_metrics_new(all_cond_score),
    }

def load_answers(filename):
    with open(filename) as f:
        data = json.load(f)
    id2answers = {d["id"]: d["answers"] for d in data}
    return id2answers

def load_and_format_predicted_answers(filename):
    with open(filename, encoding='utf-8') as f:
        output_datas = json.load(f)
    final_answer = []
    for d in output_datas:
        answer = d["answer"]
        conditions = []
        try:
            answer = d["answer"].split("Answer: ")[1].strip()
        except IndexError:
            answer = d["answer"]
        try:
            conditions_text = answer.split("Conditions: ")[1].strip()
            conditions = conditions_text.split("\n")
            # remove the conditions from the answer
            answer = answer.split(". Conditions:")[0].strip()
        except IndexError:
            conditions = []
        answer = [[answer, conditions]]
        final_answer.append({"id": d["id"], "answers": answer})
    
    return final_answer

def format_prediction(prediction, qtype):
    answer = prediction.encode('utf-8').decode('unicode_escape')
    conditions = []
    # print(prediction)
    if qtype == "yes/no" or qtype == "span":
        try:
            answer = prediction.split("Answer: ")[1].strip()
        except IndexError:
            answer = prediction
        if qtype == "yes/no" and 'yes' in answer.lower().split(' '):
            formatted_output = [["yes", []]]
        elif qtype == "yes/no" and 'no' in answer.lower().split(' '):
            formatted_output = [["no", []]]
        else:
            formatted_output = [[answer, []]]
    else:
        try:
            conditions_text = answer.split("Conditions: ")[1].strip()
            conditions = conditions_text.split("\n")
            # answer = answer.split("Conditions:")[0].strip().split("Answer:")[1].strip()
            answer_text = answer.split("Conditions:")[0].strip()
            if "Answer" in answer_text:
                answer = answer_text.split("Answer:")[1].strip()
            else:
                answer = answer.split("Conditions: ")[0].strip()
            
            if qtype == "yes/no" and 'yes' in answer.lower().split(' '):
                formatted_output = [["yes", conditions]]
            elif qtype == "yes/no" and 'no' in answer.lower().split(' '):
                formatted_output = [["no", conditions]]
            else:
                formatted_output = [[answer, conditions]]
        except IndexError:
            conditions = []
            formatted_output = [
                [answer,conditions]
            ]
    return formatted_output

def format_prediction_with_explanations(prediction, qtype):
    answer = prediction.encode('utf-8').decode('unicode_escape')
    conditions = []
    # print(prediction)
    if qtype == "yes/no" or qtype == "span":
        try:
            answer_no_exp = prediction.split("Explanation: ")[0].strip()
            answer = answer_no_exp.split("Answer: ")[1].strip()
        except IndexError:
            answer = prediction
        if qtype == "yes/no" and 'yes' in answer.lower().split(' '):
            formatted_output = [["yes", []]]
        elif qtype == "yes/no" and 'no' in answer.lower().split(' '):
            formatted_output = [["no", []]]
        else:
            formatted_output = [[answer, []]]
    else:
        try:
            conditions_text_no_exp = answer.split("Explanation: ")[0].strip()
            conditions_text = conditions_text_no_exp.split("Conditions: ")[1].strip()
            conditions = conditions_text.split("\n")
            answer = answer.split("Conditions:")[0].strip().split("Answer:")[1].strip()
            if qtype == "yes/no" and 'yes' in answer.lower().split(' '):
                formatted_output = [["yes", conditions]]
            elif qtype == "yes/no" and 'no' in answer.lower().split(' '):
                formatted_output = [["no", conditions]]
            else:
                formatted_output = [[answer, conditions]]
        except IndexError:
            conditions = []
            formatted_output = [
                [answer,conditions]
            ]
    return formatted_output

def load_ref_answers_from_predicted_file(filename):
    with open(filename, encoding='utf-8') as f:
        output_datas = [json.loads(line) for line in f]

    id2answers = {d["id"]: d["Actual_Answer"] for d in output_datas}
    id2question = {d["id"]: d["question"] for d in output_datas}
    id2qtype = {d["id"]: d["question_type"] for d in output_datas}
    # id2ans_from = {d["id"]: d["ans_from"] for d in output_datas}
    return id2answers, id2question, id2qtype

def load_and_format_predicted_answers_v2(filename):
    with open(filename, encoding='utf-8') as f:
        output_datas = [json.loads(line) for line in f]
    final_answer = []
    for d in output_datas:
        answer_text = d["answers"]
        # qtype = d['question_type']
        # clean_answer = format_prediction(answer_text, qtype.lower())
        # clean_answer = format_prediction_with_explanations(answer_text, qtype.lower())
        final_answer.append({"id": d["id"], "answers": answer_text})
        # final_answer.append({"id": d["id"], "answers": d["answer"]})
    return final_answer

def compute_metrics(prediction, reference):
    """
    Compute metrics for one example.

    args:
      prediction: a list of tuples of predicted answers and
        conditions, e.g. [(ans1, [c1, c2]), (ans2, [c3])]
      reference: same as prediction

    returns:
      A tuple of scalars for (em, em_with_conditions,
        f1, and f1_with_conditions)
    """

    # get full scores only if no answer is predicted
    if not reference:
        return [float(not prediction)] * 4

    num_answer = len(reference)

    if len(prediction) < num_answer:
        prediction.extend([("", list())] * (num_answer - len(prediction)))

    # iterate through all possible permutations
    max_em, max_f1 = 0.0, 0.0
    max_conditional_em, max_conditional_f1 = 0.0, 0.0
    for ordered_prediction in itertools.permutations(prediction):
        total_em, total_f1 = 0.0, 0.0
        total_conditional_em, total_conditional_f1 = 0.0, 0.0
        # compute metrics for one pair of answers
        for pred_answer, ref_answer in zip(ordered_prediction, reference):
            em, conditional_em, f1, conditional_f1 = compute_em_f1(
                pred_answer, ref_answer
            )
            total_em += em
            total_conditional_em += conditional_em
            total_f1 += f1
            total_conditional_f1 += conditional_f1

        # record the best permutation
        max_em = max(max_em, total_em / num_answer)
        max_conditional_em = max(max_conditional_em, total_conditional_em / num_answer)
        max_f1 = max(max_f1, total_f1 / num_answer)
        max_conditional_f1 = max(max_conditional_f1, total_conditional_f1 / num_answer)

    assert max_em <= 1 and max_f1 <= 1
    assert max_conditional_em <= 1 and max_conditional_f1 <= 1

    # discounted by extra predicted answers
    gamma = math.exp(1.0 - len(prediction) / num_answer)
    max_em *= gamma
    max_f1 *= gamma
    max_conditional_em *= gamma
    max_conditional_f1 *= gamma

    return max_em, max_conditional_em, max_f1, max_conditional_f1

def compute_em_f1(pred_answer, ref_answer):
    """
    Compute EM, F1 and with conditions for one answer.

    args:
      pred_answer: a tuple of (answer, conditions)
      ref_answer: a tuple of (answer, conditions)

    returns:
      EM, F1, and EM and F1 with conditions
    """
    # print("compute_em_f1 called")
    conditions_f1 = compute_conditions_f1(pred_answer[1], ref_answer[1])

    pred_answer_text = normalize_answer(pred_answer[0])
    ref_answer_text = normalize_answer(ref_answer[0])
    em = float(pred_answer_text == ref_answer_text)
    f1 = compute_answer_f1(ref_answer_text, pred_answer_text)
    # print("f1 = ", f1)
    # print("conditions_f1 = ", conditions_f1)
    conditional_em = em * conditions_f1
    conditions_f1 = f1 * conditions_f1
    return em, conditional_em, f1, conditions_f1

def compute_weighted_bert_metrics_V2(prediction, reference, bert_scorer):
    """
    Compute BERT F1 scores by comparing each prediction to each reference and returning the maximum score.

    Args:
      prediction: a list of tuples of predicted answers and conditions, e.g., [(ans1, [c1, c2]), (ans2, [c3])]
      reference: same as prediction
      bert_scorer: a BERT score computation object or function

    Returns:
      A tuple of scalars for (max_bert_f1, max_conditional_bert_f1)
    """
    if not reference:
        return [float(not prediction)] * 3

    max_bert_f1 = 0.0
    max_ans_conditions_bert_f1 = 0.0
    max_only_conditions_bert_f1 = 0.0

    # Compare each prediction with each reference and keep the maximum score for each metric
    for pred_answer in prediction:
        for ref_answer in reference:
            # Compute BERT scores for the current pair of prediction and reference
            bert_f1, avg_ans_conditions_bert_f1, only_conditional_bert_f1 = compute_BERTf1_V2(
                pred_answer, ref_answer, bert_scorer
            )
            
            # Update max scores for each metric
            max_bert_f1 = max(max_bert_f1, bert_f1)
            max_ans_conditions_bert_f1 = max(max_ans_conditions_bert_f1, avg_ans_conditions_bert_f1)
            max_only_conditions_bert_f1 = max(max_only_conditions_bert_f1, only_conditional_bert_f1)


    return max_bert_f1, max_ans_conditions_bert_f1, max_only_conditions_bert_f1

def compute_BERTf1_V2(pred_answer, ref_answer,bert_scorer):
    """
    Compute BERT F1, and their conditional variants for one answer.

    args:
      pred_answer: a tuple of (answer, conditions)
      ref_answer: a tuple of (answer, conditions)

    returns:
      BERT F1, conditional BERT F1
    """
    pred_answer_text = normalize_answer(pred_answer[0])
    ref_answer_text = normalize_answer(ref_answer[0])
    # print("pred_answer_text: ", pred_answer_text)
    # print("ref_answer_text: ", ref_answer_text)
    
    # Compute BERT F1 score for the answers
    P, R, bert_f1 = bert_scorer.score([pred_answer_text], [ref_answer_text])
    bert_f1 = bert_f1.item()
    
    # Compute the conditional scores (join all conditions into a single sentence)
    conditional_bert_f1 = compute_conditions_f1_V2(
        pred_answer[1], ref_answer[1],bert_scorer
    )
    
    if conditional_bert_f1 == "no_conditions":
      only_conditional_bert_f1 = 0
      avg_ans_conditions_bert_f1 = 0
        
    else:
      avg_ans_conditions_bert_f1 = (bert_f1 + conditional_bert_f1)/2.0
      only_conditional_bert_f1 = conditional_bert_f1

    return bert_f1, avg_ans_conditions_bert_f1, only_conditional_bert_f1

def compute_conditions_f1_V2(predicted_conditions, true_conditions,bert_scorer):
    """
    Compute BERT F1 scores for the set of predicted and true conditions.
    
    args:
      predicted_conditions: a list of predicted conditions
      true_conditions: a list of true conditions

    returns:
      BERT F1 for the joined conditions
    """
    if not true_conditions:
        return "no_conditions"

    if not predicted_conditions:
        return 0.0
    
    
    pred_conditions_text = " ".join(predicted_conditions)
    ref_conditions_text = " ".join(true_conditions)
    
    pred_conditions_text = normalize_answer(pred_conditions_text)
    ref_conditions_text = normalize_answer(ref_conditions_text)
    
    # Compute BERT F1 score for the conditions
    P, R, conditional_bert_f1 = bert_scorer.score([pred_conditions_text], [ref_conditions_text])
    conditional_bert_f1 = conditional_bert_f1.item()

    return conditional_bert_f1

def compute_conditions_f1(predicted_conditions, true_conditions):
    """
    Compute F1 of the predicted set of conditions.

    args:
      predicted_conditions: a list of predicted conditions
      true_conditions: a list of true conditions

    returns:
      element-wise condition F1
    """
    # print("compute_conditions_f1 called")
    if not true_conditions:
        return float(not predicted_conditions)

    if not predicted_conditions:
        return 0.0

    true_conditions = list(set(true_conditions))
    predicted_conditions = list(set(predicted_conditions))
    # print("true_conditions = ", true_conditions)
    # print("predicted_conditions = ", predicted_conditions)
    correct = sum([int(c in true_conditions) for c in predicted_conditions])
    # print("correct = ", correct)
    precision = correct / len(predicted_conditions)
    recall = correct / len(true_conditions)

    if correct == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 / (1.0 / precision + 1.0 / recall)
    # print("f1 calculaed in compute_conditions_f1 = ", f1)
    return f1

##############################################################
###################### Helper Functions ######################
##############################################################


def compute_answer_f1(a_gold, a_pred):
    """Copied from SQuAD 2.0 evaluation script."""
    # print("compute_answer_f1 called")
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    # print("inside compute ans f1 = ", f1)
    return f1


def get_tokens(s):
    """Copied from SQuAD 2.0 evaluation script."""
    if not s:
        return []
    return normalize_answer(s).split()


def normalize_answer(s):
    """Copied from SQuAD 2.0 evaluation script."""
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def parse_arguments():
    # command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file",
        dest="pred_file",
        type=str,
        default=None,
        help="Path to your prediction file.",
    )
    parser.add_argument(
        "--ref_file",
        dest="ref_file",
        type=str,
        default=None,
        help="Path to the reference file.",
    )
    parser.add_argument(
        "--output_file",
        dest="output_file",
        type=str,
        default=None,
        help="Path to the output file.",
    )
    # parser.add_argument(
    #     "--excel_file",
    #     dest="excel_file",
    #     type=str,
    #     default=None,
    #     help="Path to the excel output file.",
    # )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    results = evaluate(args.pred_file, args.ref_file, "") #, args.excel_file)
    # print(results)
    print()
    
    for key, value in results.items():
        print(key)
        for val_k, val_v in value.items():
            print(val_k, ":", val_v)
        print()

    # with open("/home/devpil/Fidelity_WI/think-on-graph/eval/ToG2/ToG2_weighted_condqa_results_newgold_2shot_autokgv3_4omini.json", "w") as f:
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    
