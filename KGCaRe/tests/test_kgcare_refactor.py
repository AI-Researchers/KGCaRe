from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kgcare.config import IndexConfig, Neo4jConfig, REPO_ROOT, slugify
from kgcare.kg_builder import find_evidence_sentence, normalize_structured_triples, parse_triplet_response, run_three_step_kg_extraction
from kgcare.llm import OpenAIChatClient
from kgcare.loaders import load_conditionalqa_questions, load_hotpotqa_questions
from kgcare.prompts import build_qa_user_prompt
from kgcare.qa import result_record
from kgcare.retriever import KGCaReTraversal
from kgcare.schemas import ClueDecision, KGStructuredTriples, PruneDecision, ReasoningDecision, RetrievalResult, StructuredQAResponse


class FakeGraphForTraversal:
    dataset = "conditionalqa"
    index_name = "pytest"

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        return [
            {
                "head_entity": "Candidate",
                "relation": "can challenge",
                "tail_entity": "Exam result",
            }
        ]


class FakeOpenAICompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAIClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=FakeOpenAICompletions(content))


def test_config_defaults_and_slugified_run_dir() -> None:
    config = IndexConfig.create(dataset="conditionalqa", index_name="dev-openai")
    assert config.embedding_model == "text-embedding-3-small"
    assert config.embedding_dimension == 1536
    assert config.index_dir == Path(__file__).resolve().parents[1] / "indexes" / "conditionalqa" / "dev-openai"
    assert slugify("mistralai/Mixtral-8x7B-Instruct-v0.1") == "mistralai-mixtral-8x7b-instruct-v0.1"
    assert config.run_dir("gpt-4.1-nano", "full").name == "full"
    hotpot_config = IndexConfig.create(dataset="hotpotqa", index_name="dev-openai")
    assert hotpot_config.dataset_paths.kg_docs_path == REPO_ROOT / "data" / "wiki_articles_supported_500"
    assert hotpot_config.dataset_paths.vector_docs_path == REPO_ROOT / "data" / "wiki_articles_supported_500"


def test_dataset_question_counts() -> None:
    conditional_questions = load_conditionalqa_questions(REPO_ROOT / "data" / "dev.json")
    hotpot_questions = load_hotpotqa_questions(REPO_ROOT / "data" / "stratified_hotpotqa_500sample_with_tag.json")
    assert len(conditional_questions) == 285
    assert len([q for q in conditional_questions if not q.not_answerable]) == 271
    assert len(hotpot_questions) == 500


def test_dataset_document_file_counts() -> None:
    assert len(list((REPO_ROOT / "data" / "docs_dev").rglob("*.txt"))) == 59
    assert len(list((REPO_ROOT / "data" / "wiki_articles_supported_500").rglob("*.txt"))) == 999
    assert len(list((REPO_ROOT / "data" / "wiki_articles_500").rglob("*.txt"))) == 4951


def test_conditionalqa_qa_prompt_comes_from_prompt_templates() -> None:
    prompt = build_qa_user_prompt(
        dataset="conditionalqa",
        question="Can I appeal?",
        vector_context="",
        kg_context="KG context:\n(Candidate, can appeal, decision)",
        qtype="yes/no_conditional",
    )
    assert "Dataset: ConditionalQA" in prompt
    assert "Question type: yes/no_conditional" in prompt
    assert "Some same-type examples are given below." in prompt
    assert "Set answer_type to \"yes_no\"" in prompt
    assert "payment plan, requires" in prompt
    assert "Candidate, can appeal, decision" in prompt


def test_conditionalqa_span_prompt_uses_span_examples() -> None:
    prompt = build_qa_user_prompt(
        dataset="conditionalqa",
        question="What is the first step?",
        vector_context="",
        kg_context="KG context:\n(first step, download and fill in, notice of appeal form)",
        qtype="span",
    )
    assert "Question type: span" in prompt
    assert "download and fill in a notice of appeal form" in prompt
    assert "Set answer_type to \"span\"" in prompt
    assert "conditions as an empty list" in prompt


def test_conditionalqa_span_conditional_prompt_uses_condition_examples() -> None:
    prompt = build_qa_user_prompt(
        dataset="conditionalqa",
        question="Which service can I use?",
        vector_context="",
        kg_context="KG context:\n(use, DBS Adult First, care home)",
        qtype="span_conditional",
    )
    assert "Question type: span_conditional" in prompt
    assert "DBS Adult First" in prompt
    assert "Put required assumptions or supporting condition statements in conditions" in prompt


def test_hotpotqa_span_prompt_forbids_yes_no_answers() -> None:
    prompt = build_qa_user_prompt(
        dataset="hotpotqa",
        question="Who was known by his stage name Aladin?",
        vector_context="",
        kg_context="KG context:\n(aladin, is stage name of, eenasul fateh)",
        qtype="span",
    )
    assert "Dataset: HotPotQA" in prompt
    assert "Question type: span" in prompt
    assert "Do not answer yes or no." in prompt
    assert 'Set answer_type to "span"' in prompt
    assert "Which film starring Emma Stone was directed by Damien Chazelle?" in prompt
    assert "(La La Land, directed by, Damien Chazelle)" in prompt


def test_hotpotqa_yesno_prompt_requires_yes_no_answers() -> None:
    prompt = build_qa_user_prompt(
        dataset="hotpotqa",
        question="Are both works novels?",
        vector_context="",
        kg_context="KG context:\n(Work A, is, novel)",
        qtype="yes/no",
    )
    assert "Question type: yes/no" in prompt
    assert 'Set answer_type to "yes_no"' in prompt
    assert 'exactly "yes" or "no"' in prompt
    assert "Are Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?" in prompt
    assert "(Scott Derrickson, nationality, American)" in prompt


def test_legacy_triplet_parser_compatibility() -> None:
    response = """
    Enhanced Triplets:
    (Student, Can challenge, GCSE exam result)
    - (Private candidate, Can appeal directly to, Awarding organisation)
    (Ignored, has, too, many, commas)
    """
    assert parse_triplet_response(response) == [
        ("Student", "Can challenge", "Gcse exam result"),
        ("Private candidate", "Can appeal directly to", "Awarding organisation"),
    ]


def test_structured_kg_output_preserves_casing_and_validates() -> None:
    payload = KGStructuredTriples.model_validate(
        {
            "triples": [
                {"subject": "HMRC", "predicate": "requires:", "object": "VAT return"},
                {"subject": "HMRC", "predicate": "requires:", "object": "VAT return"},
                {"subject": "GCSE candidate", "predicate": "can challenge", "object": "GCSE exam result"},
            ]
        }
    )
    assert normalize_structured_triples(payload) == [
        ("HMRC", "requires", "VAT return"),
        ("GCSE candidate", "can challenge", "GCSE exam result"),
    ]


def test_structured_kg_output_accepts_prompt_label_from_vllm() -> None:
    payload = KGStructuredTriples.model_validate(
        {
            "Enhanced Triplets": [
                "(Overseas Adoption, requires, General Register Office (GRO))",
                "(Adoptive parent, can apply for, adoption certificate)",
            ]
        }
    )
    assert normalize_structured_triples(payload) == [
        ("Overseas Adoption", "requires", "General Register Office (GRO)"),
        ("Adoptive parent", "can apply for", "adoption certificate"),
    ]


def test_structured_kg_output_accepts_string_triples_from_vllm() -> None:
    payload = KGStructuredTriples.model_validate(
        {
            "triples": [
                "(Asylum Applicant, must provide, documents needed for application)",
                "(Dependents, include, Children under 18)",
            ]
        }
    )
    assert normalize_structured_triples(payload) == [
        ("Asylum Applicant", "must provide", "documents needed for application"),
        ("Dependents", "include", "Children under 18"),
    ]


def test_three_step_structured_output_keeps_prompt_chain() -> None:
    class FakeLLM:
        prompts: list[str]

        def __init__(self) -> None:
            self.prompts = []

        def chat_text(self, prompt: str, max_tokens: int = 2048) -> str:
            self.prompts.append(prompt)
            return "intermediate"

        def structured(self, schema, messages, max_tokens: int = 2048):
            self.prompts.append(messages[-1]["content"])
            return schema.model_validate(
                {"triples": [{"subject": "HMRC", "predicate": "requires", "object": "VAT return"}]}
            ), '{"triples":[{"subject":"HMRC","predicate":"requires","object":"VAT return"}]}'

    llm = FakeLLM()
    _, _, raw, triples = run_three_step_kg_extraction(llm, "conditionalqa", "HMRC requires a VAT return.")
    assert len(llm.prompts) == 3
    assert "### Step 1" in llm.prompts[0]
    assert "### Step 4" in llm.prompts[1]
    assert "### Step 6" in llm.prompts[2]
    assert raw.startswith('{"triples"')
    assert triples == [("HMRC", "requires", "VAT return")]


def test_three_step_structured_output_falls_back_to_text_parser() -> None:
    class FakeLLM:
        prompts: list[str]

        def __init__(self) -> None:
            self.prompts = []

        def chat_text(self, prompt: str, max_tokens: int = 2048) -> str:
            self.prompts.append(prompt)
            if "### Step 6" in prompt:
                return "(Asylum Applicant, must provide, documents needed for application)"
            return "intermediate"

        def structured(self, schema, messages, max_tokens: int = 2048):
            self.prompts.append(messages[-1]["content"])
            raise json.JSONDecodeError("Unterminated string", doc='{"triples":["(', pos=12)

    llm = FakeLLM()
    _, _, raw, triples = run_three_step_kg_extraction(
        llm,
        "conditionalqa",
        "Asylum applicants must provide documents needed for application.",
    )
    assert len(llm.prompts) == 4
    assert "### Step 6" in llm.prompts[2]
    assert "### Step 6" in llm.prompts[3]
    assert raw == "(Asylum Applicant, must provide, documents needed for application)"
    assert triples == [("Asylum applicant", "Must provide", "Documents needed for application")]


def test_deterministic_evidence_sentence() -> None:
    text = "Applicants can appeal an exam result. HMRC requires a VAT return."
    assert find_evidence_sentence(text, ("HMRC", "requires", "VAT return")) == "HMRC requires a VAT return."


def test_structured_qa_response_serialization() -> None:
    structured = StructuredQAResponse(
        answer_type="multi",
        answers=[
            {"answer": "yes", "conditions": ["registered before the deadline"]},
            {"answer": "no", "conditions": []},
        ],
        rationale="Two conditional branches.",
    )
    assert structured.to_conditionalqa_answer() == [
        ["yes", ["registered before the deadline"]],
        ["no", []],
    ]
    assert structured.to_hotpotqa_answer() == "yes"


def test_structured_qa_response_accepts_vllm_variants() -> None:
    name_answer = StructuredQAResponse.model_validate(
        {
            "response": {
                "answer_type": "name",
                "answers": {"answer": "Eenasul Fateh"},
                "rationale": {"evidence": "Aladin is the stage name of Eenasul Fateh."},
            }
        }
    )
    assert name_answer.answer_type == "span"
    assert name_answer.to_hotpotqa_answer() == "Eenasul Fateh"
    assert "evidence" in name_answer.rationale

    text_answer = StructuredQAResponse.model_validate(
        {
            "answer_type": "text",
            "answer": "Pasek and Paul",
            "conditions": "The musical has music and lyrics written by Pasek and Paul.",
        }
    )
    assert text_answer.answer_type == "span"
    assert text_answer.to_hotpotqa_answer() == "Pasek and Paul"
    assert text_answer.to_conditionalqa_answer() == [
        ["Pasek and Paul", ["The musical has music and lyrics written by Pasek and Paul."]]
    ]

    for answer_type in ["statement", "extractive", "exact", "location"]:
        assert StructuredQAResponse.model_validate({"answer_type": answer_type, "answer": "Richmond"}).answer_type == "span"
    assert StructuredQAResponse.model_validate({"answer_type": "no_answer", "answers": []}).answer_type == "unanswerable"
    assert StructuredQAResponse.model_validate({"answer_type": "yesno", "answer": "yes"}).answer_type == "yes_no"
    assert StructuredQAResponse.model_validate({"answer_type": "yes/no_conditional", "answer": "yes"}).answer_type == "yes_no"
    assert StructuredQAResponse.model_validate({"answer_type": "conditional", "answers": []}).answer_type == "multi"


def test_openai_chat_client_json_method_uses_json_object_response_format() -> None:
    fake_client = FakeOpenAIClient(
        '{"answer_type":"span","answers":[{"answer":"Eenasul Fateh","conditions":[]}],"rationale":"ok"}'
    )
    client = OpenAIChatClient(model="local-model", api_key="test", structured_method="json")
    client.client = fake_client

    parsed, raw = client.structured(
        StructuredQAResponse,
        [{"role": "user", "content": "Return the answer."}],
        max_tokens=77,
    )

    request = fake_client.chat.completions.requests[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == 77
    assert "Return only a valid JSON object" in request["messages"][-1]["content"]
    assert parsed.to_hotpotqa_answer() == "Eenasul Fateh"
    assert raw.startswith('{"answer_type"')


def test_openai_chat_client_json_schema_method_uses_schema_response_format() -> None:
    fake_client = FakeOpenAIClient(
        '{"answer_type":"span","answers":[{"answer":"Eenasul Fateh","conditions":[]}],"rationale":"ok"}'
    )
    client = OpenAIChatClient(model="local-model", api_key="test", structured_method="json_schema")
    client.client = fake_client

    parsed, _ = client.structured(
        StructuredQAResponse,
        [{"role": "user", "content": "Return the answer."}],
        max_tokens=88,
    )

    request = fake_client.chat.completions.requests[0]
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "StructuredQAResponse"
    assert response_format["json_schema"]["schema"]["properties"]["answer_type"]["enum"] == [
        "yes_no",
        "span",
        "multi",
        "unanswerable",
    ]
    assert request["max_tokens"] == 88
    assert parsed.to_hotpotqa_answer() == "Eenasul Fateh"


def test_result_record_preserves_conditionalqa_shape() -> None:
    retrieval = RetrievalResult(
        question_id="q1",
        question="Can I appeal?",
        mode="hybrid",
        vector_context="context",
        kg_context="kg",
        triples=[["Candidate", "can challenge", "Exam result"]],
    )
    structured = StructuredQAResponse(
        answer_type="span",
        answers=[{"answer": "appeal to the awarding organisation", "conditions": ["private candidate"]}],
    )
    record = result_record(
        dataset="conditionalqa",
        question_id="q1",
        question="Can I appeal?",
        gold_answer=[["appeal", ["private candidate"]]],
        qtype="span_conditional",
        retrieval=retrieval,
        structured=structured,
        raw_response=json.dumps(structured.model_dump()),
    )
    assert record["answer"] == [["appeal to the awarding organisation", ["private candidate"]]]
    assert record["retrieval"]["triples"] == [["Candidate", "can challenge", "Exam result"]]


def test_traversal_search_prune_reasoning_flow() -> None:
    class FakeGraph:
        dataset = "conditionalqa"
        index_name = "pytest"

        def query(self, cypher: str, params: dict | None = None) -> list[dict]:
            return [
                {
                    "head_entity": "Candidate",
                    "relation": "can challenge",
                    "tail_entity": "Exam result",
                }
            ]

    class FakeLLM:
        def chat_text(self, prompt: str, *args, **kwargs) -> str:
            if "KEYWORDS:" in prompt or "extract up to" in prompt:
                return "KEYWORDS: exam"
            if "Score" in prompt and "Knowledge Triplets" in prompt:
                return "(candidate, can challenge, exam result) (Score: 10)"
            return "{Yes} {appeal to the awarding organisation}"

    traversal = KGCaReTraversal(FakeGraph(), FakeLLM(), max_depth=2, traversal_output_mode="text")
    result = traversal.retrieve("q1", "Can I challenge an exam result?")
    assert result.answer_hint == "Yes"
    assert result.triples == [["candidate", "can challenge", "exam result"]]
    assert "candidate, can challenge, exam result" in result.kg_context
    assert result.trace is not None
    assert result.trace.steps[0].partial_searches == 1


def test_structured_prune_maps_indexes_and_thresholds() -> None:
    class FakeLLM:
        def structured(self, schema, messages, max_tokens: int = 1024):
            return schema.model_validate(
                {
                    "selections": [
                        {"index": 1, "score": 9, "rationale": "directly answers"},
                        {"index": 0, "score": 7, "rationale": "weak"},
                        {"index": 99, "score": 10, "rationale": "invalid"},
                    ],
                    "rationale": "selected high-score triples",
                }
            ), "{}"

    traversal = KGCaReTraversal(FakeGraphForTraversal(), FakeLLM(), traversal_output_mode="structured")
    triples = {
        ("candidate", "can challenge", "exam result"),
        ("candidate", "has age", "18"),
    }
    selected, selected_candidates = traversal._prune_triples("Can I challenge an exam result?", triples)
    sorted_triples = sorted(triples)
    assert selected == [sorted_triples[1]]
    assert selected_candidates[0].index == 1
    assert selected_candidates[0].score == 9


def test_structured_prune_falls_back_to_legacy_parser() -> None:
    class FakeLLM:
        def structured(self, schema, messages, max_tokens: int = 1024):
            raise ValueError("bad structured output")

        def chat_text(self, prompt: str, *args, **kwargs) -> str:
            return "(candidate, can challenge, exam result) (Score: 10)"

    traversal = KGCaReTraversal(FakeGraphForTraversal(), FakeLLM(), traversal_output_mode="structured")
    selected, selected_candidates = traversal._prune_triples(
        "Can I challenge an exam result?",
        {("candidate", "can challenge", "exam result")},
    )
    assert selected == [("candidate", "can challenge", "exam result")]
    assert selected_candidates[0].score == 10


def test_structured_reasoning_sufficient_and_insufficient() -> None:
    class FakeSufficientLLM:
        def structured(self, schema, messages, max_tokens: int = 1024):
            return schema.model_validate(
                {
                    "sufficient": True,
                    "answer": "appeal to the awarding organisation",
                    "clue_entities": [],
                    "rationale": "The triple answers the question.",
                }
            ), "{}"

    traversal = KGCaReTraversal(FakeGraphForTraversal(), FakeSufficientLLM(), traversal_output_mode="structured")
    decision = traversal._reasoning(
        "Can I challenge an exam result?",
        [("candidate", "can challenge", "exam result")],
        "",
    )
    assert decision.sufficient is True
    assert decision.answer == "appeal to the awarding organisation"

    class FakeInsufficientLLM:
        def structured(self, schema, messages, max_tokens: int = 1024):
            return schema.model_validate(
                {
                    "sufficient": False,
                    "answer": "",
                    "clue_entities": ["awarding organisation", "private candidate"],
                    "rationale": "Need appeal destination details.",
                }
            ), "{}"

    traversal = KGCaReTraversal(FakeGraphForTraversal(), FakeInsufficientLLM(), traversal_output_mode="structured")
    decision = traversal._reasoning("Can I appeal?", [("candidate", "can appeal", "result")], "")
    assert decision.sufficient is False
    assert decision.clue_entities == ["awarding organisation", "private candidate"]


def test_structured_no_kg_clue_generation() -> None:
    class FakeLLM:
        def structured(self, schema, messages, max_tokens: int = 1024):
            return schema.model_validate(
                {
                    "clue_entities": ["appeal deadline", "exam board"],
                    "rationale": "Need procedural entities.",
                }
            ), "{}"

    traversal = KGCaReTraversal(FakeGraphForTraversal(), FakeLLM(), traversal_output_mode="structured")
    decision = traversal._reasoning_without_kg("When can I appeal?")
    assert decision.clue_entities == ["appeal deadline", "exam board"]


def test_text_reasoning_context_contains_full_triples() -> None:
    class FakeLLM:
        prompt = ""

        def chat_text(self, prompt: str, *args, **kwargs) -> str:
            self.prompt = prompt
            return "{Yes} {appeal}"

    llm = FakeLLM()
    traversal = KGCaReTraversal(FakeGraphForTraversal(), llm, traversal_output_mode="text")
    traversal._reasoning("Can I appeal?", [("candidate", "can challenge", "exam result")], "")
    assert "('candidate', 'can challenge', 'exam result')" in llm.prompt


def test_structured_traversal_trace() -> None:
    class FakeLLM:
        def chat_text(self, prompt: str, *args, **kwargs) -> str:
            return "KEYWORDS: exam"

        def structured(self, schema, messages, max_tokens: int = 1024):
            if schema is PruneDecision:
                return schema.model_validate(
                    {"selections": [{"index": 0, "score": 10, "rationale": "direct"}], "rationale": "keep direct"}
                ), "{}"
            if schema is ReasoningDecision:
                return schema.model_validate(
                    {
                        "sufficient": True,
                        "answer": "appeal",
                        "clue_entities": [],
                        "rationale": "enough",
                    }
                ), "{}"
            raise AssertionError(schema)

    traversal = KGCaReTraversal(FakeGraphForTraversal(), FakeLLM(), traversal_output_mode="structured")
    result = traversal.retrieve("q1", "Can I challenge an exam result?")
    assert result.answer_hint == "appeal"
    assert result.trace is not None
    assert result.trace.output_mode == "structured"
    assert result.trace.final_answer_hint == "appeal"
    assert result.trace.steps[0].candidate_triples_found == 1
    assert result.trace.steps[0].selected_triples[0].score == 10


def test_run_qa_cli_traversal_options() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "KGCaRe" / "run_qa.py"), "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--traversal-output-mode" in result.stdout
    assert "--structured-method" in result.stdout
    assert "json_schema" in result.stdout
    assert "--no-trace" in result.stdout
    assert "--qa-base-url" in result.stdout
    assert "--qa-api-key" in result.stdout
    assert "--embedding-base-url" in result.stdout
    assert "--embedding-api-key" in result.stdout


def test_faiss_vector_metadata_alignment(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    import numpy as np

    try:
        faiss.swig_ptr(np.ascontiguousarray(np.array([[1.0, 0.0, 0.0]], dtype="float32")))
    except ValueError as exc:
        pytest.skip(f"local FAISS/NumPy ABI cannot accept NumPy arrays: {exc}")

    from kgcare.schemas import ChunkRecord
    from kgcare.vector_store import FaissChunkStore

    class FakeEmbeddingClient:
        def embed_texts(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] if "alpha" in text else [0.0, 1.0, 0.0] for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    chunks = [
        ChunkRecord(
            chunk_id="c1",
            doc_id="d1",
            dataset="conditionalqa",
            title="Alpha",
            text="alpha text",
            source_path="alpha.txt",
        ),
        ChunkRecord(
            chunk_id="c2",
            doc_id="d2",
            dataset="conditionalqa",
            title="Beta",
            text="beta text",
            source_path="beta.txt",
        ),
    ]
    store = FaissChunkStore(tmp_path / "faiss.index", tmp_path / "vector_metadata.jsonl", FakeEmbeddingClient())
    store.build(chunks, dimension=3)
    store.load()
    assert store.count == 2
    hits = store.search("alpha", top_k=1)
    assert hits[0][0].chunk_id == "c1"


@pytest.mark.skipif(os.getenv("KGCARE_TEST_NEO4J") != "1", reason="requires a running Neo4j test database")
def test_neo4j_namespace_upsert_and_stats() -> None:
    from kgcare.graph_store import Neo4jTripleStore
    from kgcare.schemas import TripleRecord

    graph = Neo4jTripleStore(Neo4jConfig(), "conditionalqa", "pytest-refactor")
    try:
        graph.reset_namespace()
        graph.upsert_triples(
            [
                TripleRecord(
                    triple_id="t1",
                    dataset="conditionalqa",
                    index_name="pytest-refactor",
                    chunk_id="c1",
                    doc_id="d1",
                    head="Candidate",
                    relation="can challenge",
                    tail="Exam result",
                )
            ]
        )
        assert graph.stats() == {"nodes": 2, "edges": 1}
        assert graph.export_triples(limit=1) == [("Candidate", "can challenge", "Exam result")]
    finally:
        graph.reset_namespace()
        graph.close()
