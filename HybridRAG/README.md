# HybridRAG Run Guide

## Files

```text
prepare_conditionalqa.py       Prepare ConditionalQA JSONL files
prepare_hotpotqa.py            Prepare HotpotQA JSONL files
build_index.py                 Build ConditionalQA index
build_hotpotqa_index.py        Build HotpotQA index
run_hybrid.py                  Run ConditionalQA QA
run_hotpotqa.py                Run HotpotQA QA
eval_conditionalqa.py          Evaluate ConditionalQA output
eval_hotpotqa.py               Evaluate HotpotQA output
run_matrix.sh                  Run ConditionalQA matrix
hybridrag/                     Python package used by the scripts
```

## Install

Run from the repository root:

```bash
python -m pip install -r HybridRAG/requirements.txt
```

## Configure

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
```

For local OpenAI-compatible models:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=not_needed
```

## Default Models

```text
KG construction: gpt-4o-2024-08-06
QA:              gpt-3.5-turbo-0125
Embeddings:      text-embedding-3-small
```

## Prepare ConditionalQA Data

```bash
python HybridRAG/prepare_conditionalqa.py
```

## Build ConditionalQA Index

```bash
python HybridRAG/build_index.py \
  --kg-model gpt-4o-2024-08-06 \
  --embedding-model text-embedding-3-small
```

Smoke-test index:

```bash
python HybridRAG/build_index.py \
  --kg-model gpt-4o-2024-08-06 \
  --embedding-model text-embedding-3-small \
  --limit-docs 5 \
  --force-vector \
  --force-kg
```

## Run ConditionalQA

```bash
python HybridRAG/run_hybrid.py \
  --provider openai \
  --llm-model gpt-3.5-turbo-0125 \
  --run-name full
```

Smoke test:

```bash
python HybridRAG/run_hybrid.py \
  --provider openai \
  --llm-model gpt-3.5-turbo-0125 \
  --run-name smoke \
  --limit 10 \
  --overwrite
```

Local OpenAI-compatible model:

```bash
python HybridRAG/run_hybrid.py \
  --provider vllm \
  --llm-model local-qa-model \
  --llm-base-url http://127.0.0.1:8050/v1 \
  --llm-api-key not_needed \
  --run-name local-smoke \
  --limit 10 \
  --overwrite
```

## Evaluate ConditionalQA

```bash
python HybridRAG/eval_conditionalqa.py \
  --input HybridRAG/adapter_runs/conditionalqa/openai/gpt-3-5-turbo-0125/full/predictions.jsonl
```

## Prepare HotpotQA Data

```bash
python HybridRAG/prepare_hotpotqa.py
```

## Build HotpotQA Index

```bash
python HybridRAG/build_hotpotqa_index.py \
  --kg-provider vllm \
  --kg-model local-kg-model \
  --kg-base-url http://127.0.0.1:8000/v1 \
  --kg-api-key not_needed \
  --embedding-model text-embedding-3-small \
  --structured-method json
```

## Run HotpotQA

```bash
python HybridRAG/run_hotpotqa.py \
  --provider openai \
  --llm-model gpt-3.5-turbo-0125 \
  --run-name full
```

## Evaluate HotpotQA

```bash
python HybridRAG/eval_hotpotqa.py \
  --input HybridRAG/adapter_runs/hotpotqa/openai/gpt-3-5-turbo-0125/full/predictions.jsonl
```

## Output Paths

```text
HybridRAG/adapter_data/<dataset>/
HybridRAG/indexes/<dataset>/
HybridRAG/adapter_runs/<dataset>/<provider>/<model>/<run-name>/
```

Run files:

```text
predictions.jsonl
run_config.json
results.json
```

## Main Options

Index:

```text
--index-dir PATH
--kg-model MODEL
--kg-base-url URL
--kg-api-key KEY
--embedding-model MODEL
--limit-docs N
--force-vector
--force-kg
--structured-method function_calling|json
```

QA:

```text
--provider openai|openai-compatible|vllm
--llm-model MODEL
--llm-base-url URL
--llm-api-key KEY
--run-name NAME
--limit N
--overwrite
--vector-fetch-k N
--vector-context-k N
--graph-seed-k N
--graph-depth N
--graph-context-k N
```

## Matrix Run

```bash
bash HybridRAG/run_matrix.sh
```
