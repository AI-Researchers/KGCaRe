import os
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _parse_conditionalqa_answer(answer: str) -> tuple[str, list[str]]:
    answer = answer.replace("\r", "").strip()
    conditions: list[str] = []

    answer_match = re.search(r"Answer:\s*(.*?)(?:\nConditions:|$)", answer, flags=re.S | re.I)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        answer_text = answer.strip()

    cond_match = re.search(r"Conditions:\s*(.*)$", answer, flags=re.S | re.I)
    if cond_match:
        cond_text = cond_match.group(1).strip()
        for line in cond_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower() == "none":
                continue
            if line.startswith("-") or line.startswith("*"):
                line = line[1:].strip()
            conditions.append(line)

    return answer_text, conditions


def _extract_json_obj(answer: str) -> Optional[dict]:
    answer = answer.strip()
    start = answer.find("{")
    if start == -1:
        return None

    for end in range(len(answer), start, -1):
        candidate = answer[start:end]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _parse_conditionalqa_json_answer(answer: str) -> tuple[str, list[str]]:
    parsed = _extract_json_obj(answer)
    if not parsed:
        raise ValueError("No valid JSON object found")

    answer_text = str(parsed.get("Answer", parsed.get("answer", ""))).strip()
    conditions = parsed.get("Conditions", parsed.get("conditions", []))

    if isinstance(conditions, str):
        normalized_conditions: list[str] = []
        for line in conditions.splitlines():
            line = line.strip()
            if not line or line.lower() == "none":
                continue
            if line.startswith("-") or line.startswith("*"):
                line = line[1:].strip()
            normalized_conditions.append(line)
        conditions = normalized_conditions
    elif isinstance(conditions, list):
        conditions = [str(c).strip() for c in conditions if str(c).strip()]
    else:
        conditions = [str(conditions).strip()]

    return answer_text, conditions


def format_prediction(prediction: str, qtype: str, dataset: str = "conditionalqa"):
    """
    Parse model output to extract formatted answer and conditions.

    Args:
        prediction (str): Raw model output from LLM.
        qtype (str): Question type (yes/no, span).
        dataset (str): Dataset name, e.g., "conditionalqa" or "hotpotqa".

    Returns:
        List[List[str, List[str]]]: [answer, [conditions]]
    """
    answer = prediction.encode("utf-8").decode("unicode_escape").strip()
    conditions: list[str] = []

    print(f"[LLM Raw Output]\n{prediction}\n")

    # ---------- HOTPOTQA ----------
    if dataset == "hotpotqa":
        # Normalize yes/no
        if qtype == "yes/no":
            normalized = answer.lower()
            if "yes" == normalized or "yes" in normalized.split():
                return "yes"
            elif "no" == normalized or "no" in normalized.split():
                return "no"
        # Otherwise treat as span
        return answer

    # ---------- CONDITIONALQA ----------
    if qtype in ("yes/no", "span"):
        try:
            answer_text, conditions = _parse_conditionalqa_json_answer(answer)
        except Exception:
            answer_text, conditions = _parse_conditionalqa_answer(answer)

        if qtype == "yes/no":
            normalized = answer_text.lower().split()
            if "yes" in normalized:
                return [["yes", []]]
            elif "no" in normalized:
                return [["no", []]]
        return [[answer_text, []]]

    try:
        answer_text, conditions = _parse_conditionalqa_json_answer(answer)
    except Exception:
        answer_text, conditions = _parse_conditionalqa_answer(answer)

    if qtype == "yes/no":
        normalized = answer_text.lower().split()
        if "yes" in normalized:
            return [["yes", conditions]]
        elif "no" in normalized:
            return [["no", conditions]]
    return [[answer_text, conditions]]


def load_processed_ids(jsonl_path: str) -> set:
    """
    Return a set of IDs already processed in the output file.

    Args:
        jsonl_path (str): Path to existing output file

    Returns:
        set: Set of processed question IDs
    """
    if not os.path.exists(jsonl_path):
        return set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        return {json.loads(line.strip())["id"] for line in f if line.strip()}
