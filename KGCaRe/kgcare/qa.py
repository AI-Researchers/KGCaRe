from __future__ import annotations

from typing import Any, Dict, List

from kgcare.llm import OpenAIChatClient
from kgcare.prompts import QA_SYSTEM_PROMPT, build_qa_user_prompt
from kgcare.schemas import RetrievalResult, StructuredQAResponse


class KGCaReQA:
    def __init__(self, llm: OpenAIChatClient, max_tokens: int = 1024) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def answer(self, dataset: str, question: str, retrieval: RetrievalResult, qtype: str = "span") -> tuple[StructuredQAResponse, str]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_qa_user_prompt(
                    dataset=dataset,
                    question=question,
                    vector_context=retrieval.vector_context,
                    kg_context=retrieval.kg_context,
                    qtype=qtype,
                ),
            },
        ]
        try:
            return self.llm.structured(StructuredQAResponse, messages, max_tokens=self.max_tokens)
        except Exception as exc:
            fallback = StructuredQAResponse(
                answer_type="unanswerable",
                answers=[],
                rationale=f"Structured QA parsing failed: {exc}",
            )
            return fallback, ""


def result_record(
    dataset: str,
    question_id: str,
    question: str,
    gold_answer: Any,
    qtype: str,
    retrieval: RetrievalResult,
    structured: StructuredQAResponse,
    raw_response: str,
) -> dict[str, Any]:
    if dataset == "conditionalqa":
        answer: Any = structured.to_conditionalqa_answer()
    else:
        answer = structured.to_hotpotqa_answer()
    return {
        "id": question_id,
        "_id": question_id,
        "question": question,
        "question_type": qtype,
        "answer": answer,
        "predicted_answer": answer,
        "gold_answer": gold_answer,
        "retrieval": retrieval.model_dump(),
        "raw_response": raw_response,
        "structured_response": structured.model_dump(),
    }
