# KGCaRe

`KGCaRe/` contains the main implementation of the proposed method used in the paper artifact. It builds a document-level knowledge graph, retrieves context through KG traversal and vector search, and generates structured answers for ConditionalQA and the sampled HotpotQA setting.

KGCaRe is the method we compare against the retained baselines. The baseline adapter folders are useful for reproducing comparisons, but this folder is the primary implementation.

## What KGCaRe Does

The pipeline has four main steps:

1. Build a knowledge graph from source documents.
2. Build a FAISS vector index over source text chunks.
3. Retrieve KG context and vector context for each question.
4. Generate and evaluate structured answers.

The KG and vector channels are deliberately separate until final answer generation. The KG channel supports symbolic traversal over extracted triples. The vector channel preserves source-text wording, broader context, and exact span conditions that may be lost during triple extraction.

## Implementation Map

```text
KGCaRe/build_index.py              Build KG triples, Neo4j graph, FAISS vectors, and index manifest
KGCaRe/run_qa.py                   Run QA with hybrid, KG-only, vector-only, or no-context modes
KGCaRe/inspect_index.py            Inspect local index files and Neo4j graph counts
KGCaRe/eval_conditionalqa.py       Evaluate ConditionalQA predictions
KGCaRe/eval_hotpotqa.py            Evaluate HotpotQA predictions
KGCaRe/kgcare/config.py            Dataset paths, index paths, model defaults, Neo4j config
KGCaRe/kgcare/kg_builder.py        Document loading, prompt-chain KG extraction, triple normalization
KGCaRe/kgcare/graph_store.py       Neo4j storage for extracted triples
KGCaRe/kgcare/vector_store.py      FAISS vector index over normalized embeddings
KGCaRe/kgcare/retriever.py         KG traversal, pruning, reasoning, and hybrid retrieval wrapper
KGCaRe/kgcare/qa.py                Final structured answer generation
KGCaRe/kgcare/prompts.py           KG construction, traversal, reasoning, and QA prompts
KGCaRe/tests/                      Focused regression tests for the refactored implementation
```

## Data Used By This Artifact

The default paths are defined in `kgcare/config.py`.

ConditionalQA:

- Documents: `data/docs_dev/*.txt`
- Questions: `data/dev.json`
- Original JSON files retained under `data/conditionalqa/`

HotpotQA sample:

- Documents: `data/wiki_articles_supported_500/*.txt`
- Questions: `data/stratified_hotpotqa_500sample_with_tag.json`

Generated indexes and run outputs are not committed. They are written under `KGCaRe/indexes/` and `KGCaRe/adapter_runs/`, both of which should remain local artifacts.

## Environment

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r KGCaRe/requirements.txt
```

Required services and credentials:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1

export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=...
export NEO4J_DATABASE=neo4j
```

`OPENAI_BASE_URL` is optional for the normal OpenAI API. Set it when using an OpenAI-compatible endpoint. Neo4j 5 is required for KG storage and traversal.

## Model Defaults

The artifact defaults are:

- KG construction: `gpt-4o-2024-08-06`
- QA: `gpt-3.5-turbo-0125`
- Embeddings: `text-embedding-3-small`

You can override these from the command line. For local vLLM or another OpenAI-compatible endpoint, pass `--kg-base-url`, `--qa-base-url`, and `--*-api-key not_needed`.

## Build An Index

ConditionalQA:

```bash
python KGCaRe/build_index.py \
  --dataset conditionalqa \
  --index-name conditionalqa-gpt4o \
  --kg-model gpt-4o-2024-08-06 \
  --embedding-model text-embedding-3-small \
  --reset-graph \
  --overwrite-vectors \
  --overwrite-triples
```

HotpotQA sample:

```bash
python KGCaRe/build_index.py \
  --dataset hotpotqa \
  --index-name hotpotqa-gpt4o \
  --kg-model gpt-4o-2024-08-06 \
  --embedding-model text-embedding-3-small \
  --reset-graph \
  --overwrite-vectors \
  --overwrite-triples
```

Useful build options:

- `--limit-docs N`: build a smoke-test index from the first `N` documents.
- `--kg-output-mode structured`: use structured output for final triple extraction.
- `--kg-output-mode text`: use text parsing fallback behavior.
- `--index-root PATH`: write index files outside the default `KGCaRe/indexes/`.
- `--reset-graph`: remove existing triples for the same dataset/index namespace before ingesting.
- `--overwrite-vectors`: rebuild FAISS vectors even if files already exist.
- `--overwrite-triples`: rerun KG extraction and rewrite `triples.jsonl`.

The index directory contains:

```text
chunks.jsonl              Source chunk metadata
triples.jsonl             Extracted triples and evidence snippets
faiss.index               FAISS IndexFlatIP vector index
vector_metadata.jsonl     Text and metadata for each vector row
manifest.json             Build configuration and counts
```

Triples are also inserted into Neo4j with `dataset` and `index_name` namespace properties so multiple indexes can coexist.

## Inspect An Index

```bash
python KGCaRe/inspect_index.py \
  --dataset conditionalqa \
  --index-name conditionalqa-gpt4o \
  --sample-triples 10
```

This reports JSONL counts, FAISS vector counts/dimensions, Neo4j graph counts, manifest metadata, and a small triple sample.

## Run QA

Hybrid KG + vector retrieval:

```bash
python KGCaRe/run_qa.py \
  --dataset conditionalqa \
  --index-name conditionalqa-gpt4o \
  --mode hybrid \
  --qa-model gpt-3.5-turbo-0125 \
  --run-name full \
  --traversal-output-mode structured \
  --structured-method json_schema
```

HotpotQA:

```bash
python KGCaRe/run_qa.py \
  --dataset hotpotqa \
  --index-name hotpotqa-gpt4o \
  --mode hybrid \
  --qa-model gpt-3.5-turbo-0125 \
  --run-name full \
  --traversal-output-mode structured \
  --structured-method json_schema
```

Retrieval modes:

- `hybrid`: retrieve both KG traversal context and vector passages.
- `kg`: use only KG traversal context.
- `vector`: use only vector context.
- `no_context`: send the question without retrieved context.

Useful QA options:

- `--limit N`: run a smoke test on the first `N` questions.
- `--overwrite`: replace an existing `output.jsonl`.
- `--vector-top-k N`: number of vector chunks passed into the prompt.
- `--kg-max-depth N`: maximum KG traversal depth.
- `--kg-max-entities N`: maximum entities retained during traversal.
- `--traversal-output-mode structured`: keep traversal evidence in structured form.
- `--structured-method json_schema`: request JSON-schema-compatible structured QA output.
- `--no-trace`: omit traversal traces from predictions to reduce output size.

Outputs are written to:

```text
KGCaRe/adapter_runs/<dataset>/<provider>/<qa-model-slug>/<run-name>/
  output.jsonl
  run_config.json
  results.json              created after evaluation
```

Each prediction row includes the question id, question text, gold answer, normalized prediction, raw model response, structured response, retrieval context, and optional traversal trace.

## Evaluate

ConditionalQA:

```bash
python KGCaRe/eval_conditionalqa.py \
  --run-dir KGCaRe/adapter_runs/conditionalqa/openai/gpt-3.5-turbo-0125/full
```

HotpotQA:

```bash
python KGCaRe/eval_hotpotqa.py \
  --run-dir KGCaRe/adapter_runs/hotpotqa/openai/gpt-3.5-turbo-0125/full
```

You can also pass a prediction file directly:

```bash
python KGCaRe/eval_conditionalqa.py \
  --pred-file KGCaRe/adapter_runs/conditionalqa/openai/gpt-3.5-turbo-0125/full/output.jsonl
```

The evaluator writes `results.json` next to the prediction file.

## KG Construction Details

The KG builder uses a three-stage prompt chain:

1. Contextual fact extraction.
2. Conditional and exception expansion.
3. Logical refinement and triple normalization.

The final output is normalized into `(subject, predicate, object)` triples. Each triple is assigned a deterministic id and linked to its source chunk and best-matching evidence sentence. Structured output is used when possible; text parsing is retained as a fallback for weaker or local models.

## Retrieval Details

KG retrieval starts from topic entities supplied by the dataset when available, or from keywords extracted from the question. At depth 1, the traversal searches existing KG heads, tails, and relation labels with partial matching. At later depths, it follows selected candidate entities. The LLM is used to prune candidate triples and decide whether the current evidence is sufficient. If it is not sufficient, the model can propose clue entities for the next traversal step.

Vector retrieval uses `text-embedding-3-small` by default. Vectors are L2-normalized and searched with `faiss.IndexFlatIP`, which gives exact inner-product search over normalized vectors.

## Local vLLM Example

Build with a local OpenAI-compatible KG model:

```bash
python KGCaRe/build_index.py \
  --dataset conditionalqa \
  --index-name conditionalqa-local \
  --kg-model Qwen3.6-27B \
  --kg-base-url http://127.0.0.1:8000/v1 \
  --kg-api-key not_needed \
  --embedding-model text-embedding-3-small \
  --reset-graph \
  --overwrite-vectors \
  --overwrite-triples
```

Run QA with a local OpenAI-compatible model:

```bash
python KGCaRe/run_qa.py \
  --dataset conditionalqa \
  --index-name conditionalqa-local \
  --mode kg \
  --provider vllm \
  --qa-model mistralai/Mistral-7B-Instruct-v0.3 \
  --qa-base-url http://127.0.0.1:8050/v1 \
  --qa-api-key not_needed \
  --run-name mistral-kg-smoke \
  --limit 10 \
  --overwrite
```

## Common Problems

- Missing Neo4j password: set `NEO4J_PASSWORD`; the artifact does not hardcode local credentials.
- Missing FAISS files during QA: run `build_index.py` first with the same `--dataset` and `--index-name`.
- Empty KG traversal: inspect the graph with `inspect_index.py`; also check that `--index-name` matches the index used during build.
- Structured output parse failures: try `--structured-method json` for local models, reduce `--max-tokens`, or run a smaller smoke test first.
- Slow indexing: KG construction calls the LLM per document chunk, so full builds can take time and API budget.

## Tests

Focused regression tests:

```bash
python -m pytest KGCaRe/tests/test_kgcare_refactor.py
```

Two tests may be skipped in a local environment: one if the installed FAISS/NumPy ABI is incompatible with the vector test, and one if no Neo4j test database is running.
