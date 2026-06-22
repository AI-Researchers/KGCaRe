# KGCaRe Run Guide

## Files

```text
build_index.py              Build KG triples, Neo4j graph data, FAISS vectors, and manifest files
run_qa.py                   Run QA with hybrid, KG-only, vector-only, or no-context modes
inspect_index.py            Print index counts and sample triples
eval_conditionalqa.py       Evaluate ConditionalQA predictions
eval_hotpotqa.py            Evaluate HotpotQA predictions
kgcare/                     Python package used by the scripts
tests/                      Regression tests
```

## Install

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r KGCaRe/requirements.txt
```

## Configure

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1

export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=...
export NEO4J_DATABASE=neo4j
```

Neo4j must be running before building or querying a KG index.

## Default Models

```text
KG construction: gpt-4o-2024-08-06
QA:              gpt-3.5-turbo-0125
Embeddings:      text-embedding-3-small
```

## Build ConditionalQA Index

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

Smoke-test build:

```bash
python KGCaRe/build_index.py \
  --dataset conditionalqa \
  --index-name conditionalqa-smoke \
  --kg-model gpt-4o-2024-08-06 \
  --embedding-model text-embedding-3-small \
  --limit-docs 5 \
  --reset-graph \
  --overwrite-vectors \
  --overwrite-triples
```

## Build HotpotQA Index

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

## Build With A Local OpenAI-Compatible KG Model

```bash
python KGCaRe/build_index.py \
  --dataset conditionalqa \
  --index-name conditionalqa-local \
  --kg-model local-kg-model \
  --kg-base-url http://127.0.0.1:8000/v1 \
  --kg-api-key not_needed \
  --embedding-model text-embedding-3-small \
  --reset-graph \
  --overwrite-vectors \
  --overwrite-triples
```

## Index Output

Default path:

```text
KGCaRe/indexes/<dataset>/<index-name>/
```

Files:

```text
chunks.jsonl
triples.jsonl
faiss.index
vector_metadata.jsonl
manifest.json
```

## Inspect An Index

```bash
python KGCaRe/inspect_index.py \
  --dataset conditionalqa \
  --index-name conditionalqa-gpt4o \
  --sample-triples 10
```

## Run ConditionalQA

Hybrid retrieval:

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

KG-only:

```bash
python KGCaRe/run_qa.py \
  --dataset conditionalqa \
  --index-name conditionalqa-gpt4o \
  --mode kg \
  --qa-model gpt-3.5-turbo-0125 \
  --run-name kg-only \
  --traversal-output-mode structured \
  --structured-method json_schema
```

Vector-only:

```bash
python KGCaRe/run_qa.py \
  --dataset conditionalqa \
  --index-name conditionalqa-gpt4o \
  --mode vector \
  --qa-model gpt-3.5-turbo-0125 \
  --run-name vector-only
```

Smoke test:

```bash
python KGCaRe/run_qa.py \
  --dataset conditionalqa \
  --index-name conditionalqa-gpt4o \
  --mode hybrid \
  --qa-model gpt-3.5-turbo-0125 \
  --run-name smoke \
  --limit 10 \
  --overwrite
```

## Run HotpotQA

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

## Run With A Local OpenAI-Compatible QA Model

```bash
python KGCaRe/run_qa.py \
  --dataset conditionalqa \
  --index-name conditionalqa-local \
  --mode kg \
  --provider vllm \
  --qa-model local-qa-model \
  --qa-base-url http://127.0.0.1:8050/v1 \
  --qa-api-key not_needed \
  --run-name local-kg \
  --limit 10 \
  --overwrite
```

## QA Output

Default path:

```text
KGCaRe/adapter_runs/<dataset>/<provider>/<qa-model>/<run-name>/
```

Files:

```text
output.jsonl
run_config.json
results.json
```

`results.json` is created by the evaluation scripts.

## Evaluate ConditionalQA

```bash
python KGCaRe/eval_conditionalqa.py \
  --run-dir KGCaRe/adapter_runs/conditionalqa/openai/gpt-3.5-turbo-0125/full
```

Or:

```bash
python KGCaRe/eval_conditionalqa.py \
  --pred-file KGCaRe/adapter_runs/conditionalqa/openai/gpt-3.5-turbo-0125/full/output.jsonl
```

## Evaluate HotpotQA

```bash
python KGCaRe/eval_hotpotqa.py \
  --run-dir KGCaRe/adapter_runs/hotpotqa/openai/gpt-3.5-turbo-0125/full
```

Or:

```bash
python KGCaRe/eval_hotpotqa.py \
  --pred-file KGCaRe/adapter_runs/hotpotqa/openai/gpt-3.5-turbo-0125/full/output.jsonl
```

## Main Options

Build:

```text
--dataset conditionalqa|hotpotqa
--index-name NAME
--kg-model MODEL
--kg-base-url URL
--kg-api-key KEY
--embedding-model MODEL
--limit-docs N
--index-root PATH
--reset-graph
--overwrite-vectors
--overwrite-triples
```

QA:

```text
--mode hybrid|kg|vector|no_context
--qa-model MODEL
--qa-base-url URL
--qa-api-key KEY
--run-name NAME
--limit N
--overwrite
--vector-top-k N
--kg-max-depth N
--kg-max-entities N
--traversal-output-mode structured|text
--structured-method parse|json|json_schema
```

## Tests

```bash
python -m pytest KGCaRe/tests/test_kgcare_refactor.py
```
