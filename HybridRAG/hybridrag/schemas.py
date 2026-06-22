from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


EntityType = Literal[
    "person",
    "organization",
    "location",
    "place",
    "country",
    "city",
    "date",
    "time",
    "work",
    "creative_work",
    "event",
    "award",
    "sports_team",
    "institution",
    "concept",
    "object",
    "occupation",
    "benefit",
    "service",
    "person_group",
    "condition",
    "requirement",
    "exception",
    "deadline",
    "amount",
    "document",
    "action",
    "process",
    "rule",
    "other",
]


class RefinedChunk(BaseModel):
    """Compact factual notes produced by the first KG creation step."""

    summary: str = Field(description="A compact faithful summary of the chunk.")
    key_facts: list[str] = Field(default_factory=list, description="Atomic facts preserved from the chunk.")
    exact_evidence_snippets: list[str] = Field(
        default_factory=list,
        description="Exact source snippets, preferably preserving HTML tags when present.",
    )
    entities: list[str] = Field(default_factory=list, description="Important entities mentioned in the chunk.")
    section_path: str | None = Field(default=None, description="Best available heading/section path.")


class KGTriple(BaseModel):
    """One validated graph triple extracted from a refined document chunk."""

    head: str = Field(description="Concise subject entity or rule.")
    head_type: EntityType = Field(description="Semantic type for the head.")
    relation: str = Field(description="Normalized relation phrase.")
    tail: str = Field(description="Concise object entity, condition, outcome, date, amount, or requirement.")
    tail_type: EntityType = Field(description="Semantic type for the tail.")
    evidence: str = Field(description="Exact source text supporting the triple.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Document and chunk metadata.")

    @field_validator("head", "relation", "tail", "evidence")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field must not be empty")
        return value

    @field_validator("head_type", "tail_type", mode="before")
    @classmethod
    def normalize_entity_type(cls, value: str) -> str:
        allowed = {
            "person",
            "organization",
            "location",
            "place",
            "country",
            "city",
            "date",
            "time",
            "work",
            "creative_work",
            "event",
            "award",
            "sports_team",
            "institution",
            "concept",
            "object",
            "occupation",
            "benefit",
            "service",
            "person_group",
            "condition",
            "requirement",
            "exception",
            "deadline",
            "amount",
            "document",
            "action",
            "process",
            "rule",
            "other",
        }
        normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        return normalized if normalized in allowed else "other"


class KGExtractionResult(BaseModel):
    """Structured result for the second KG creation step."""

    triples: list[KGTriple] = Field(default_factory=list)


class YesNoAnswer(BaseModel):
    answer_type: Literal["yes_no"]
    answer: Literal["yes", "no"]
    conditions: list[str] = Field(default_factory=list)
    rationale: str | None = None


class SpanAnswer(BaseModel):
    answer_type: Literal["span"]
    answer: str = Field(description="Shortest exact answer span copied from context.")
    conditions: list[str] = Field(default_factory=list)
    rationale: str | None = None

    @field_validator("answer")
    @classmethod
    def span_answer_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Span answer must not be empty")
        return value


class MultiAnswer(BaseModel):
    answer_type: Literal["multi"]
    answers: list[YesNoAnswer | SpanAnswer] = Field(description="Multiple answer entries.")
    rationale: str | None = None


class UnanswerableAnswer(BaseModel):
    answer_type: Literal["unanswerable"]
    answer: Literal[""] = ""
    conditions: list[str] = Field(default_factory=list)
    rationale: str | None = None


ConditionalQAResponse = Annotated[
    Union[YesNoAnswer, SpanAnswer, MultiAnswer, UnanswerableAnswer],
    Field(discriminator="answer_type"),
]
ConditionalQAResponseAdapter = TypeAdapter(ConditionalQAResponse)


class ConditionalQAResponseEnvelope(BaseModel):
    """Wrapper used for LangChain structured output compatibility."""

    response: ConditionalQAResponse

    model_config = ConfigDict(extra="forbid")


class HotpotQAAnswer(BaseModel):
    """Structured short answer used for HotPotQA QA runs."""

    answer: str = Field(default="", description="Final short answer string.")
    rationale: str | None = Field(default=None, description="Optional short rationale for debugging.")

    model_config = ConfigDict(extra="forbid")


def response_to_eval_answers(response: ConditionalQAResponse) -> list[list[Any]]:
    """Convert structured answer variants to ConditionalQA eval shape."""

    if isinstance(response, UnanswerableAnswer):
        return []

    if isinstance(response, YesNoAnswer):
        return [[response.answer, clean_conditions(response.conditions)]]

    if isinstance(response, SpanAnswer):
        return [[response.answer, clean_conditions(response.conditions)]]

    normalized: list[list[Any]] = []
    for item in response.answers:
        normalized.extend(response_to_eval_answers(item))
    return normalized


def clean_conditions(conditions: list[str]) -> list[str]:
    return [condition.strip() for condition in conditions if condition and condition.strip()]


def normalize_eval_answers_for_question_type(
    answers: list[list[Any]],
    question_type: str,
) -> list[list[Any]]:
    """Keep model output aligned with ConditionalQA's expected answer family."""

    if not answers:
        return []

    normalized: list[list[Any]] = []
    keep_conditions = question_type.endswith("_conditional")
    for answer, conditions in answers:
        answer_text = str(answer).strip()
        if question_type.startswith("yes/no"):
            lowered = answer_text.lower()
            if lowered.startswith("yes"):
                answer_text = "yes"
            elif lowered.startswith("no"):
                answer_text = "no"
        condition_list = clean_conditions(list(conditions or [])) if keep_conditions else []
        normalized.append([answer_text, condition_list])
    return normalized


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from model text."""

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_answer_payload(payload: dict[str, Any]) -> ConditionalQAResponse:
    if "response" in payload:
        return ConditionalQAResponseEnvelope.model_validate(payload).response
    return ConditionalQAResponseAdapter.validate_python(payload)
