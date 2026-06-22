from __future__ import annotations

import json

from hybridrag.schemas import ConditionalQAResponseEnvelope, HotpotQAAnswer, KGExtractionResult, RefinedChunk

ENTITY_TYPE_HINT = (
    "person, organization, location, place, country, city, date, time, work, creative_work, "
    "event, award, sports_team, institution, concept, object, occupation, benefit, service, "
    "person_group, condition, requirement, exception, deadline, amount, document, action, "
    "process, rule, other"
)


REFINE_SYSTEM_PROMPT = """You convert source text into compact factual notes for knowledge-graph extraction.
Do not add facts. Do not infer beyond the text. Preserve exact wording for named entities, dates,
locations, affiliations, works, roles, relationships, conditions, exceptions, amounts, and actions."""


TRIPLE_SYSTEM_PROMPT = """You extract a knowledge graph from refined source text. Output only validated triples.
A triple must be directly supported by the refined text or exact evidence snippets."""


QA_SYSTEM_PROMPT = """You answer ConditionalQA questions using only the supplied HybridRAG context.
Return only the requested structured answer. Do not use outside knowledge.
Keep the JSON small, complete, and syntactically valid."""

HOTPOTQA_SYSTEM_PROMPT = """You answer HotPotQA questions using only the supplied HybridRAG context.
Return only the requested structured answer. Do not use outside knowledge.
Keep the JSON small, complete, and syntactically valid."""


def build_refine_messages(
    *,
    title: str,
    url: str,
    chunk_id: str,
    section_path: str,
    chunk_text: str,
) -> list[dict[str, str]]:
    user_prompt = f"""Document title: {title}
Document URL: {url}
Chunk ID: {chunk_id}
Section path: {section_path}

Raw chunk:
{chunk_text}

Return structured notes that:
- keep all factual claims, named entities, relationships, dates, places, roles, rules, and exceptions
- preserve exact evidence snippets when important facts or relationships are stated
- resolve local pronouns only when the referent is explicit in the chunk
- keep source snippets intact when useful for later exact-answer evaluation
- omit navigation text, duplicated boilerplate, and irrelevant page furniture

Return only one JSON object with this exact instance shape, not a JSON schema:
{{
  "summary": "one compact faithful summary of this chunk",
  "key_facts": ["atomic fact copied or faithfully condensed from the chunk"],
  "exact_evidence_snippets": ["exact supporting source text"],
  "entities": ["important entity name"],
  "section_path": "{section_path or title}"
}}

All keys are required. Arrays may be empty. Do not return keys such as "$defs",
"properties", "required", "title", "type", or "description".
"""
    return [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_triple_messages(
    *,
    title: str,
    url: str,
    chunk_id: str,
    section_path: str,
    refined_chunk_json: str,
) -> list[dict[str, str]]:
    user_prompt = f"""Document title: {title}
Document URL: {url}
Chunk ID: {chunk_id}
Section path: {section_path}

Refined notes:
{refined_chunk_json}

Extract triples using this meaning:
- head: concise entity or rule subject
- relation: normalized verb phrase such as "is", "was born in", "located in", "part of",
  "member of", "created by", "performed by", "winner of", "date is", "amount is",
  "is eligible if", "must", "cannot", "requires", "applies to", "has exception"
- tail: concise entity, attribute, condition, outcome, date, amount, location, or requirement
- evidence: exact supporting source text, preferably copied from original snippets
- metadata: document title, URL, chunk ID, section path

Rules:
- Do not create triples from unsupported inference.
- Keep entities concise, normally under 6 words.
- Preserve exact evidence text.
- Include exceptions, negative rules, and comparison facts as explicit triples.
- For yes/no rules, make eligibility and prohibition triples explicit.
- Use only these entity type labels: {ENTITY_TYPE_HINT}.
- Prefer triples that connect named entities, dates, locations, creators, teams, roles, awards,
  works, and direct attributes.
- Keep evidence to the shortest exact sentence or clause that supports the triple.

Return only one JSON object with this exact instance shape, not a JSON schema:
{{
  "triples": [
    {{
      "head": "concise subject entity",
      "head_type": "person",
      "relation": "normalized relation phrase",
      "tail": "concise object entity or value",
      "tail_type": "other",
      "evidence": "exact supporting source text",
      "metadata": {{}}
    }}
  ]
}}

All keys are required. The triples array may be empty only when the chunk has no
extractable factual relationships. Do not return keys such as "$defs",
"properties", "required", "title", "type", or "description".
"""
    return [
        {"role": "system", "content": TRIPLE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_qa_messages(
    *,
    question_type: str,
    scenario: str,
    question: str,
    vector_context: str,
    graph_context: str,
) -> list[dict[str, str]]:
    if question_type.endswith("_conditional"):
        condition_rule = (
            "- this is a conditional question: include only exact condition snippets copied from context.\n"
            "- condition strings must be source text/evidence snippets, not graph triple notation.\n"
        )
    else:
        condition_rule = "- this is not a conditional question: every conditions field must be [].\n"

    user_prompt = f"""Question type: {question_type}
Scenario:
{scenario}

Question:
{question}

Vector context:
{vector_context}

Graph triple context:
{graph_context}

Answer rules:
- yes/no questions: answer exactly "yes" or "no".
- span questions: copy the shortest exact answer span from context.
- multi-answer questions: include only the answer entries needed to answer the question, with at most 5 entries.
- if the answer is not supported by context, return unanswerable.
- keep each answer concise; do not paste long paragraphs when a shorter exact span answers the question.
- keep each conditions list concise; include at most 5 condition strings per answer.
- answers and conditions must not include graph triple notation such as "head:type -[relation]-> tail:type".
{condition_rule.rstrip()}
- return an object matching this wrapper schema: {{"response": <one answer variant>}}.
- close every JSON object and array; no comments, markdown, or trailing text.

Structured output schema:
{json.dumps(ConditionalQAResponseEnvelope.model_json_schema(), indent=2)}
"""
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_hotpotqa_messages(
    *,
    question_type: str,
    question: str,
    vector_context: str,
    graph_context: str,
) -> list[dict[str, str]]:
    user_prompt = f"""Question type: {question_type}

Question:
{question}

Vector context:
{vector_context}

Graph triple context:
{graph_context}

Answer rules:
- answer using only the vector context and graph triple context.
- for yes/no questions, answer exactly "yes" or "no".
- for span questions, output the shortest final answer span, not a full sentence.
- for bridge/comparison questions, combine facts across the retrieved contexts when needed.
- if the answer is not supported by context, return an empty answer string.
- do not include graph triple notation in the answer.
- return an object matching this schema: {{"answer": "final short answer", "rationale": "optional short note"}}.
- close every JSON object; no comments, markdown, or trailing text.

Structured output schema:
{json.dumps(HotpotQAAnswer.model_json_schema(), indent=2)}
"""
    return [
        {"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
