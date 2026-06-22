# LightRAG Baseline Adapter

`lightRAG/adapters/` contains local adapter code for running the upstream LightRAG implementation on the datasets used in the KGCaRe paper artifact. The upstream LightRAG package is kept mostly isolated; this folder handles data conversion, run configuration, structured answer normalization, and evaluation bridges.

This is a baseline adapter, not the KGCaRe implementation.

## Role In The Artifact

LightRAG is included as a comparison system for graph-augmented retrieval. The adapter makes its input and output format compatible with the artifact datasets and evaluators. It does not reproduce KGCaRe's Neo4j traversal, pruning, clue-entity search, or hybrid prompt construction.

## Implementation Map

```text
lightRAG/adapters/common.py                     Shared paths and JSONL helpers
lightRAG/adapters/output_schemas.py            Structured output schemas
lightRAG/adapters/conditionalqa_adapter.py      Build ConditionalQA corpus/query JSONL
lightRAG/adapters/hotpotqa_adapter.py           Build HotpotQA corpus/query JSONL
lightRAG/adapters/run_conditionalqa_vllm.py     Run LightRAG on ConditionalQA
lightRAG/adapters/run_hotpotqa_vllm.py          Run LightRAG on HotpotQA
lightRAG/adapters/eval_conditionalqa_bridge.py  Convert/score ConditionalQA outputs
lightRAG/adapters/eval_hotpotqa_bridge.py       Convert/score HotpotQA outputs
```

## Data Flow

The adapter first converts repository data into JSONL files expected by the runner scripts.

ConditionalQA:

```text
data/docs_dev/*.txt
data/dev.json
  -> lightRAG/adapter_data/conditionalqa/corpus.jsonl
  -> lightRAG/adapter_data/conditionalqa/queries.jsonl
```

HotpotQA:

```text
data/wiki_articles_supported_500/*.txt
data/stratified_hotpotqa_500sample_with_tag.json
  -> lightRAG/adapter_data/hotpotqa/corpus.jsonl
  -> lightRAG/adapter_data/hotpotqa/queries.jsonl
```

LightRAG writes its generated graph/vector workspace under `lightRAG/workspaces/` by default. Predictions are written under `lightRAG/adapter_runs/`. These generated folders are ignored by Git.

## Environment

Install LightRAG dependencies from the `lightRAG/` directory. The exact dependency set depends on whether you use the included upstream package or an editable install.

```bash
cd lightRAG
python -m pip install -e .
python -m pip install -r requirements-offline.txt
```

For OpenAI runs:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
```

For local OpenAI-compatible LLM and embedding servers:

```bash
export LLM_BINDING_HOST=http://127.0.0.1:8000/v1
export LLM_BINDING_API_KEY=not_needed
export EMBEDDING_BINDING_HOST=http://127.0.0.1:8001/v1
export EMBEDDING_BINDING_API_KEY=not_needed
export EMBEDDING_DIM=1024
```

If using OpenAI embeddings with a local LLM, pass `--embedding-model text-embedding-3-small`, `--embedding-base-url https://api.openai.com/v1`, and an OpenAI key.

## Prepare Adapter Data

Run from the `lightRAG/` directory:

```bash
python adapters/conditionalqa_adapter.py
python adapters/hotpotqa_adapter.py
```

The scripts are idempotent and rewrite the adapter JSONL files.

## ConditionalQA Runs

OpenAI model and OpenAI embeddings:

```bash
python adapters/run_conditionalqa_vllm.py \
  --provider openai \
  --llm-model gpt-3.5-turbo-0125 \
  --embedding-model text-embedding-3-small \
  --ingest \
  --limit 10
```

Local OpenAI-compatible model:

```bash
python adapters/run_conditionalqa_vllm.py \
  --provider openai-compatible \
  --llm-model mistralai/Mistral-7B-Instruct-v0.3 \
  --llm-base-url http://127.0.0.1:8050/v1 \
  --llm-api-key not_needed \
  --embedding-model text-embedding-3-small \
  --embedding-base-url https://api.openai.com/v1 \
  --ingest \
  --limit 10
```

Important options:

- `--ingest`: build/update the LightRAG workspace from the corpus.
- `--limit N`: smoke-test on the first `N` queries.
- `--working-dir PATH`: override the LightRAG workspace directory.
- `--max-total-tokens N`: cap context passed to the LLM.
- `--no-structured-output`: disable the adapter's structured answer prompt.
- `--json-enforcement`: pass the Pydantic schema as response format during QA. Do not use with `--ingest`.

## HotpotQA Runs

OpenAI model and embeddings:

```bash
python adapters/run_hotpotqa_vllm.py \
  --provider openai \
  --llm-model gpt-3.5-turbo-0125 \
  --embedding-model text-embedding-3-small \
  --ingest \
  --limit 10
```

Local model with file-based graph storage:

```bash
python adapters/run_hotpotqa_vllm.py \
  --provider openai-compatible \
  --llm-model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --llm-base-url http://127.0.0.1:8051/v1 \
  --llm-api-key not_needed \
  --embedding-model text-embedding-3-small \
  --embedding-base-url https://api.openai.com/v1 \
  --graph-storage NetworkXStorage \
  --ingest \
  --limit 10
```

HotpotQA-specific options:

- `--parallel-insert N`: control ingestion parallelism.
- `--ingest-only`: build the workspace and skip QA. Must be used with `--ingest`.
- `--graph-storage NetworkXStorage`: use local file-based graph storage.
- `--graph-storage Neo4JStorage`: use Neo4j-backed graph storage when configured.
- `--no-llm-cache`: disable LightRAG's LLM cache. This is important for multi-model comparisons because cache reuse can otherwise hide model differences.

## Evaluation Bridges

ConditionalQA:

```bash
python adapters/eval_conditionalqa_bridge.py \
  --input adapter_runs/conditionalqa/predictions.jsonl
```

HotpotQA:

```bash
python adapters/eval_hotpotqa_bridge.py \
  --input adapter_runs/hotpotqa/predictions.jsonl
```

If `--input` is omitted, each bridge uses its default prediction path from `common.py`.

## Output Layout

Typical generated paths:

```text
lightRAG/adapter_data/<dataset>/
  corpus.jsonl
  queries.jsonl
  metadata.json

lightRAG/workspaces/<dataset-or-run-name>/
  LightRAG internal graph/vector/cache files

lightRAG/adapter_runs/<dataset>/
  predictions.jsonl
  converted_predictions.jsonl
  results.json
```

The exact internal workspace files are controlled by upstream LightRAG and may change between upstream versions.

## Notes For Paper Reviewers

- This adapter uses LightRAG's own retrieval and storage logic.
- The adapter adds structured output parsing only for evaluation compatibility.
- ConditionalQA span answers are post-processed to keep exact spans when the model returns valid JSON.
- HotpotQA answers are normalized into a short-answer format suitable for EM/F1.
- For multi-model experiments, rebuild or isolate workspaces carefully and disable LLM cache when comparing QA models.

## Common Problems

- Import errors: install the upstream LightRAG package from the `lightRAG/` directory.
- Empty predictions: run the data adapter first and check the generated `queries.jsonl`.
- Stale answers across model runs: use a fresh workspace or pass `--no-llm-cache` for HotpotQA.
- Embedding dimension mismatch: pass `--embedding-dim` to match the embedding model or server.
- Guided JSON errors with local models: avoid `--json-enforcement` and rely on prompt-based structured output.
