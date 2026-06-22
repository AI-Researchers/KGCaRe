# HippoRAG Adapter Run Guide

## Files

```text
common.py                         Shared path and JSONL helpers
conditionalqa_adapter.py          Prepare ConditionalQA JSONL files
hotpotqa_adapter.py               Prepare HotpotQA JSONL files
run_utils.py                      Shared indexing and QA wrapper
run_conditionalqa_openai.py       Run ConditionalQA with OpenAI
run_conditionalqa_vllm.py         Run ConditionalQA with a local endpoint
run_hotpotqa_openai.py            Run HotpotQA with OpenAI
run_hotpotqa_vllm.py              Run HotpotQA with a local endpoint
eval_conditionalqa_bridge.py      Evaluate ConditionalQA predictions
eval_hotpotqa_bridge.py           Evaluate HotpotQA predictions
run_matrix_smoke.sh               Run a small smoke matrix
```

## Install

Run from the repository root:

```bash
cd HippoRAG
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Configure

For OpenAI:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

For a local OpenAI-compatible endpoint:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8050/v1
export VLLM_API_KEY=not_needed
```

## Prepare Data

Run from `HippoRAG/`:

```bash
python adapters/conditionalqa_adapter.py
python adapters/hotpotqa_adapter.py
```

Output:

```text
adapter_data/conditionalqa/
adapter_data/hotpotqa/
```

The run scripts can also prepare data with `--prepare-data`.

## Run ConditionalQA With OpenAI

```bash
python adapters/run_conditionalqa_openai.py \
  --llm-model gpt-3.5-turbo-0125 \
  --embedding-model text-embedding-3-small \
  --prepare-data \
  --ingest \
  --run-name full \
  --limit 20
```

## Run ConditionalQA With A Local Model

```bash
python adapters/run_conditionalqa_vllm.py \
  --llm-model local-qa-model \
  --llm-base-url http://127.0.0.1:8050/v1 \
  --embedding-model text-embedding-3-small \
  --embedding-base-url https://api.openai.com/v1 \
  --prepare-data \
  --ingest \
  --run-name local-smoke \
  --limit 20
```

## Evaluate ConditionalQA

```bash
python adapters/eval_conditionalqa_bridge.py \
  --input adapter_runs/conditionalqa/openai/gpt-3-5-turbo-0125/full/predictions.jsonl
```

## Run HotpotQA With OpenAI

```bash
python adapters/run_hotpotqa_openai.py \
  --llm-model gpt-3.5-turbo-0125 \
  --embedding-model text-embedding-3-small \
  --prepare-data \
  --ingest \
  --run-name full \
  --limit 20
```

## Run HotpotQA With A Local Model

```bash
python adapters/run_hotpotqa_vllm.py \
  --llm-model local-qa-model \
  --llm-base-url http://127.0.0.1:8051/v1 \
  --embedding-model text-embedding-3-small \
  --embedding-base-url https://api.openai.com/v1 \
  --prepare-data \
  --ingest \
  --run-name local-smoke \
  --limit 20
```

## Shared HotpotQA Index

Build index only:

```bash
python adapters/run_hotpotqa_openai.py \
  --llm-model gpt-4o-2024-08-06 \
  --embedding-model text-embedding-3-small \
  --prepare-data \
  --ingest \
  --index-only \
  --index-dir adapter_runs/hotpotqa/index-gpt4o
```

Run QA with that index:

```bash
python adapters/run_hotpotqa_openai.py \
  --llm-model gpt-3.5-turbo-0125 \
  --embedding-model text-embedding-3-small \
  --index-dir adapter_runs/hotpotqa/index-gpt4o \
  --run-name full
```

## Evaluate HotpotQA

```bash
python adapters/eval_hotpotqa_bridge.py \
  --input adapter_runs/hotpotqa/openai/gpt-3-5-turbo-0125/full/predictions.jsonl
```

## Output Paths

Prepared data:

```text
adapter_data/<dataset>/
```

Runs:

```text
adapter_runs/<dataset>/<provider>/<model>/<run-name>/
```

Run files:

```text
predictions.jsonl
run_config.json
output.jsonl
eval_predictions.jsonl
eval_predictions.json
results.json
```

## Main Options

```text
--llm-model MODEL
--llm-base-url URL
--embedding-model MODEL
--embedding-base-url URL
--run-name NAME
--limit N
--docs-dir PATH
--ref-file PATH
--prepare-data
--ingest
--index-only
--index-dir PATH
--force-index-from-scratch
--force-openie-from-scratch
--retrieval-top-k N
--qa-top-k N
--max-new-tokens N
--temperature FLOAT
--output-path PATH
```

## Smoke Matrix

```bash
bash adapters/run_matrix_smoke.sh
```
