from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DatasetName = Literal["conditionalqa", "hotpotqa"]
RetrievalMode = Literal["hybrid", "kg", "vector", "no_context"]
TraversalOutputMode = Literal["structured", "text"]


class SourceDocument(BaseModel):
    doc_id: str
    dataset: DatasetName
    title: str
    text: str
    source_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    dataset: DatasetName
    title: str
    text: str
    source_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0


class TripleRecord(BaseModel):
    triple_id: str
    dataset: DatasetName
    index_name: str
    chunk_id: str
    doc_id: str
    head: str
    relation: str
    tail: str
    source_path: str = ""
    evidence: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def as_tuple(self) -> tuple[str, str, str]:
        return self.head, self.relation, self.tail


class VectorMetadata(BaseModel):
    vector_id: int
    chunk_id: str
    doc_id: str
    dataset: DatasetName
    title: str
    source_path: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IndexManifest(BaseModel):
    dataset: DatasetName
    index_name: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    kg_model: str
    kg_output_mode: Literal["text", "structured"] = "structured"
    embedding_model: str
    embedding_dimension: int
    neo4j_uri: str
    neo4j_database: str
    document_count: int = 0
    chunk_count: int = 0
    vector_count: int = 0
    triple_count: int = 0
    node_count: Optional[int] = None
    edge_count: Optional[int] = None
    files: Dict[str, str] = Field(default_factory=dict)


class KGStructuredTriple(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str

    @field_validator("subject", "predicate", "object")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Triple fields must be non-empty")
        return value


class KGStructuredTriples(BaseModel):
    model_config = ConfigDict(extra="forbid")

    triples: List[KGStructuredTriple]

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_triplet_keys(cls, data: Any) -> Any:
        """Accept vLLM JSON that follows the prompt label instead of schema key."""
        if not isinstance(data, dict):
            return data
        if "triples" in data:
            return {"triples": cls._coerce_legacy_triplets(data["triples"])}
        for key in ["Enhanced Triplets", "Enhanced Triples", "enhanced_triplets", "triplets", "Triplets"]:
            if key in data:
                return {"triples": cls._coerce_legacy_triplets(data[key])}
        return data

    @classmethod
    def _coerce_legacy_triplets(cls, value: Any) -> List[Dict[str, str]]:
        items = value if isinstance(value, list) else [value]
        triples: List[Dict[str, str]] = []
        for item in items:
            triples.extend(cls._coerce_legacy_triplet_item(item))
        return triples

    @classmethod
    def _coerce_legacy_triplet_item(cls, item: Any) -> List[Dict[str, str]]:
        if isinstance(item, dict):
            subject = item.get("subject") or item.get("head") or item.get("s")
            predicate = item.get("predicate") or item.get("relation") or item.get("p")
            obj = item.get("object") or item.get("tail") or item.get("o")
            if subject and predicate and obj:
                return [{"subject": str(subject), "predicate": str(predicate), "object": str(obj)}]
            return []
        if isinstance(item, (list, tuple)) and len(item) == 3:
            return [{"subject": str(item[0]), "predicate": str(item[1]), "object": str(item[2])}]
        if not isinstance(item, str):
            return []

        parsed: List[Dict[str, str]] = []
        for line in item.splitlines() or [item]:
            text = line.strip().lstrip("-*0123456789. ")
            if "(" in text and ")" in text and text.index("(") < text.rindex(")"):
                text = text[text.index("(") + 1 : text.rindex(")")]
            parts = [part.strip() for part in text.split(",")]
            if len(parts) == 3 and all(parts):
                parsed.append({"subject": parts[0], "predicate": parts[1], "object": parts[2]})
        return parsed


class ConditionalQAQuestion(BaseModel):
    id: str
    scenario: str
    question: str
    answers: List[List[Any]] = Field(default_factory=list)
    not_answerable: bool = False
    url: str = ""
    evidences: List[str] = Field(default_factory=list)

    @property
    def query_text(self) -> str:
        return f"{self.scenario} {self.question}".strip()


class HotpotQAQuestion(BaseModel):
    id: str = Field(alias="_id")
    question: str
    answer: str = ""
    qtype: str = "span"
    entities: List[str] = Field(default_factory=list)

    @property
    def query_text(self) -> str:
        return self.question


class TripleCandidate(BaseModel):
    index: int
    triple: List[str]
    score: Optional[float] = None
    rationale: str = ""

    @field_validator("triple")
    @classmethod
    def valid_triple(cls, value: List[str]) -> List[str]:
        if len(value) != 3:
            raise ValueError("Triple must contain exactly three strings")
        cleaned = [part.strip() for part in value]
        if any(not part for part in cleaned):
            raise ValueError("Triple fields must be non-empty")
        return cleaned


class PrunedTripleSelection(BaseModel):
    index: int
    score: float = Field(ge=0, le=10)
    rationale: str


class PruneDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selections: List[PrunedTripleSelection]
    rationale: str


class ReasoningDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    answer: str
    clue_entities: List[str]
    rationale: str

    @field_validator("answer")
    @classmethod
    def clean_answer(cls, value: str) -> str:
        return value.strip()

    @field_validator("clue_entities")
    @classmethod
    def clean_clues(cls, value: List[str]) -> List[str]:
        return [item.strip() for item in value if item.strip()]


class ClueDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clue_entities: List[str]
    rationale: str

    @field_validator("clue_entities")
    @classmethod
    def clean_clues(cls, value: List[str]) -> List[str]:
        return [item.strip() for item in value if item.strip()]


class TraversalStepTrace(BaseModel):
    depth: int
    topic_entities: List[str] = Field(default_factory=list)
    clue_entities: List[str] = Field(default_factory=list)
    searched_topic_entities: List[str] = Field(default_factory=list)
    searched_clue_entities: List[str] = Field(default_factory=list)
    partial_searches: int = 0
    exact_searches: int = 0
    clue_searches: int = 0
    candidate_triples_found: int = 0
    selected_triples: List[TripleCandidate] = Field(default_factory=list)
    next_topic_entities: List[str] = Field(default_factory=list)
    next_clue_entities: List[str] = Field(default_factory=list)
    reasoning_sufficient: Optional[bool] = None
    reasoning_answer: str = ""
    reasoning_rationale: str = ""
    stop_reason: str = ""


class TraversalTrace(BaseModel):
    output_mode: TraversalOutputMode
    initial_entities: List[str] = Field(default_factory=list)
    steps: List[TraversalStepTrace] = Field(default_factory=list)
    final_answer_hint: str = ""


class RetrievalResult(BaseModel):
    question_id: str
    question: str
    mode: RetrievalMode
    vector_context: str = ""
    kg_context: str = ""
    vector_chunk_ids: List[str] = Field(default_factory=list)
    triples: List[List[str]] = Field(default_factory=list)
    answer_hint: str = ""
    trace: Optional[TraversalTrace] = None


class QAAnswerItem(BaseModel):
    answer: str
    conditions: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_answer_item(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"answer": data, "conditions": []}
        if isinstance(data, dict):
            cleaned = dict(data)
            if "answer" not in cleaned:
                for key in ["text", "value", "response", "name"]:
                    if key in cleaned:
                        cleaned["answer"] = cleaned[key]
                        break
            conditions = cleaned.get("conditions", [])
            if isinstance(conditions, str):
                cleaned["conditions"] = [conditions] if conditions.strip() else []
            return cleaned
        return data

    @field_validator("answer", mode="before")
    @classmethod
    def coerce_answer_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    @field_validator("conditions", mode="before")
    @classmethod
    def coerce_conditions(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []


class StructuredQAResponse(BaseModel):
    answer_type: Literal["yes_no", "span", "multi", "unanswerable"] = "span"
    answers: List[QAAnswerItem] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_model_variants(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "response" in data and isinstance(data["response"], dict):
            data = data["response"]
        cleaned = dict(data)
        answer_type = str(cleaned.get("answer_type", "span")).strip().lower()
        answer_type_aliases = {
            "yes/no": "yes_no",
            "yes-no": "yes_no",
            "yesno": "yes_no",
            "boolean": "yes_no",
            "yesno_conditional": "yes_no",
            "yes_no_conditional": "yes_no",
            "yes/no_conditional": "yes_no",
            "conditional": "multi",
            "no_answer": "unanswerable",
            "no answer": "unanswerable",
            "none": "unanswerable",
            "not_answerable": "unanswerable",
            "name": "span",
            "text": "span",
            "string": "span",
            "entity": "span",
            "number": "span",
            "date": "span",
            "statement": "span",
            "extractive": "span",
            "exact": "span",
            "location": "span",
        }
        cleaned["answer_type"] = answer_type_aliases.get(answer_type, answer_type or "span")

        if "answers" not in cleaned and "answer" in cleaned:
            cleaned["answers"] = [{"answer": cleaned["answer"], "conditions": cleaned.get("conditions", [])}]
        answers = cleaned.get("answers", [])
        if isinstance(answers, dict):
            cleaned["answers"] = [answers[key] for key in sorted(answers, key=str)]
        elif isinstance(answers, str):
            cleaned["answers"] = [{"answer": answers, "conditions": []}]

        rationale = cleaned.get("rationale", "")
        if not isinstance(rationale, str):
            cleaned["rationale"] = json.dumps(rationale, ensure_ascii=False)
        return cleaned

    def to_conditionalqa_answer(self) -> List[List[Any]]:
        if not self.answers:
            return [["", []]]
        return [[item.answer, item.conditions] for item in self.answers]

    def to_hotpotqa_answer(self) -> str:
        if not self.answers:
            return ""
        return self.answers[0].answer


def read_jsonl_models(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    records: list[BaseModel] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(model.model_validate_json(line))
    return records


def write_jsonl_models(path: Path, records: list[BaseModel], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
