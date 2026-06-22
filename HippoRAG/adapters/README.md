# HippoRAG Baseline Adapter

`HippoRAG/adapters/` contains local adapter code for running upstream HippoRAG on the KGCaRe artifact datasets. It prepares ConditionalQA and HotpotQA data, runs HippoRAG indexing and QA, and converts predictions into formats compatible with the artifact evaluators.

This is a baseline adapter. It is not the proposed KGCaRe method.

## Role In The Artifact

HippoRAG is included as a graph-based retrieval baseline. It builds its own graph and vector stores using the upstream HippoRAG implementation. The adapter keeps dataset preparation, output paths, and evaluation format consistent with the KGCaRe experiments.

The key comparison point is that HippoRAG uses its own graph construction and retrieval pipeline, while KGCaRe uses a Neo4j triple store plus iterative KG traversal, triple pruning, clue-entity expansion, and final answer generation over separately retrieved KG/vector context.

## Implementation Map

```text
HippoRAG/adapters/common.py                     Shared path and JSONL helpers
HippoRAG/adapters/conditionalqa_adapter.py      Build ConditionalQA corpus/query JSONL
HippoRAG/adapters/hotpotqa_adapter.py           Build HotpotQA corpus/query JSONL
HippoRAG/adapters/run_utils.py                  Shared HippoRAG indexing and QA wrapper
HippoRAG/adapters/run_conditionalqa_openai.py   Run ConditionalQA with OpenAI models
HippoRAG/adapters/run_conditionalqa_vllm.py     Run ConditionalQA with OpenAI-compatible models
HippoRAG/adapters/run_hotpotqa_openai.py        Run HotpotQA with OpenAI models
HippoRAG/adapters/run_hotpotqa_vllm.py          Run HotpotQA with OpenAI-compatible models
HippoRAG/adapters/eval_conditionalqa_bridge.py  Convert/score ConditionalQA predictions
HippoRAG/adapters/eval_hotpotqa_bridge.py       Convert/score HotpotQA predictions
HippoRAG/adapters/run_matrix_smoke.sh           Small smoke-test matrix
```

## Data Flow

ConditionalQA:

```text
data/docs_dev/*.txt
data/dev.json
  -> HippoRAG/adapter_data/conditionalqa/corpus.jsonl
  -> HippoRAG/adapter_data/conditionalqa/queries.jsonl
```

HotpotQA:

```text
data/wiki_articles_supported_500/*.txt
data/stratified_hotpotqa_500sample_with_tag.json
  -> HippoRAG/adapter_data/hotpotqa/corpus.jsonl
  -> HippoRAG/adapter_data/hotpotqa/queries.jsonl
```

HippoRAG indexes and predictions are written under `HippoRAG/adapter_runs/`. These are generated artifacts and are ignored by Git.

## Environment

Install HippoRAG dependencies from the `HippoRAG/` directory:

```bash
cd HippoRAG
python -m pip install -e .
python -m pip install -r requirements.txt
```

For OpenAI models and embeddings:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

For local OpenAI-compatible models:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8050/v1
export VLLM_API_KEY=not_needed
```

The adapter sets `OPENAI_API_KEY=not_needed` automatically when a local base URL is used and no OpenAI key is present, because parts of the upstream stack expect an API-key-like value.

## Prepare Adapter Data

From `HippoRAG/`:

```bash
python adapters/conditionalqa_adapter.py
python adapters/hotpotqa_adapter.py
```

Alternatively, the run scripts can prepare data when passed `--prepare-data`.

## ConditionalQA Runs

OpenAI:

```bash
python adapters/run_conditionalqa_openai.py \
  --llm-model gpt-3.5-turbo-0125 \
  --embedding-model text-embedding-3-small \
  --prepare-data \
  --ingest \
  --run-name full \
  --limit 20
```

Local OpenAI-compatible model:

```bash
python adapters/run_conditionalqa_vllm.py \
  --llm-model mistralai/Mistral-7B-Instruct-v0.3 \
  --llm-base-url http://127.0.0.1:8050/v1 \
  --embedding-model text-embedding-3-small \
  --embedding-base-url https://api.openai.com/v1 \
  --prepare-data \
  --ingest \
  --run-name mistral-smoke \
  --limit 20
```

Useful options:

- `--prepare-data`: regenerate adapter JSONL files before running.
- `--ingest`: build or update the HippoRAG index before QA.
- `--force-index-from-scratch`: rebuild the full index.
- `--force-openie-from-scratch`: rerun OpenIE extraction.
- `--retrieval-top-k`: number of retrieved items from HippoRAG retrieval.
- `--qa-top-k`: number of retrieved items passed into QA.
- `--max-new-tokens`: maximum answer generation tokens.
- `--temperature`: model temperature.

## HotpotQA Runs

OpenAI:

```bash
python adapters/run_hotpotqa_openai.py \
  --llm-model gpt-3.5-turbo-0125 \
  --embedding-model text-embedding-3-small \
  --prepare-data \
  --ingest \
  --run-name full \
  --limit 20
```

Local OpenAI-compatible model:

```bash
python adapters/run_hotpotqa_vllm.py \
  --llm-model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --llm-base-url http://127.0.0.1:8051/v1 \
  --embedding-model text-embedding-3-small \
  --embedding-base-url https://api.openai.com/v1 \
  --prepare-data \
  --ingest \
  --run-name mixtral-smoke \
  --limit 20
```

Build a shared HotpotQA index and skip QA:

```bash
python adapters/run_hotpotqa_openai.py \
  --llm-model gpt-4o-2024-08-06 \
  --embedding-model text-embedding-3-small \
  --prepare-data \
  --ingest \
  --index-only \
  --index-dir adapter_runs/hotpotqa/index-gpt4o
```

Then reuse the shared index for a QA run:

```bash
python adapters/run_hotpotqa_openai.py \
  --llm-model gpt-3.5-turbo-0125 \
  --embedding-model text-embedding-3-small \
  --index-dir adapter_runs/hotpotqa/index-gpt4o \
  --run-name full
```

## Evaluation Bridges

ConditionalQA:

```bash
python adapters/eval_conditionalqa_bridge.py \
  --input adapter_runs/conditionalqa/openai/gpt-3-5-turbo-0125/full/predictions.jsonl
```

HotpotQA:

```bash
python adapters/eval_hotpotqa_bridge.py \
  --input adapter_runs/hotpotqa/openai/gpt-3-5-turbo-0125/full/predictions.jsonl
```

The bridge scripts convert HippoRAG output into normalized answer records and then call the artifact evaluation helpers.

## Output Layout

```text
HippoRAG/adapter_data/<dataset>/
  corpus.jsonl
  queries.jsonl
  metadata.json

HippoRAG/adapter_runs/<dataset>/<provider>/<model-slug>/<run-name>/
  predictions.jsonl
  run_config.json
  results.json
  HippoRAG internal graph/vector/OpenIE files
```

When `--index-dir` is provided, index files can live outside the QA run directory so multiple QA models can share the same index.

## Smoke Matrix

```bash
bash adapters/run_matrix_smoke.sh
```

This helper runs small ConditionalQA and HotpotQA checks. It is intended to verify wiring, not to reproduce full paper numbers.

## Notes For Paper Reviewers

- HippoRAG indexing can be expensive because OpenIE extraction and graph construction are model-driven.
- `--ingest` is needed for the first run or whenever the index should be rebuilt.
- For fair QA comparisons, reuse the same index across QA models where possible.
- The adapter preserves raw HippoRAG responses and retrieved documents for inspection.
- Evaluation bridges are adapter code; the upstream HippoRAG internals are not modified for KGCaRe.

## Common Problems

- Missing adapter JSONL files: run the data adapter scripts or pass `--prepare-data`.
- Reusing stale graph/OpenIE files: pass `--force-index-from-scratch` and `--force-openie-from-scratch`.
- Local endpoint credential errors: pass a local base URL and use `not_needed` as the API key.
- Slow first run: build a small smoke test with `--limit 20` before starting a full run.
- Output path confusion: check `run_config.json`; it records dataset, provider, model, run name, index directory, and prediction path.
