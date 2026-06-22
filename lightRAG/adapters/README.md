# LightRAG Adapter Run Guide

## Files

```text
common.py                       Shared path and JSONL helpers
conditionalqa_adapter.py        Prepare ConditionalQA JSONL files
hotpotqa_adapter.py             Prepare HotpotQA JSONL files
run_conditionalqa_vllm.py       Run ConditionalQA
run_hotpotqa_vllm.py            Run HotpotQA
eval_conditionalqa_bridge.py    Evaluate ConditionalQA predictions
eval_hotpotqa_bridge.py         Evaluate HotpotQA predictions
output_schemas.py               Structured answer schemas
```

## Install

Run from the repository root:

```bash
cd lightRAG
python -m pip install -e .
python -m pip install -r requirements-offline.txt
```

## Configure

For OpenAI:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
```

For local OpenAI-compatible servers:

```bash
export LLM_BINDING_HOST=http://127.0.0.1:8000/v1
export LLM_BINDING_API_KEY=not_needed
export EMBEDDING_BINDING_HOST=http://127.0.0.1:8001/v1
export EMBEDDING_BINDING_API_KEY=not_needed
export EMBEDDING_DIM=1024
```

## Prepare Data

Run from `lightRAG/`:

```bash
python adapters/conditionalqa_adapter.py
python adapters/hotpotqa_adapter.py
```

Output:

```text
adapter_data/conditionalqa/
adapter_data/hotpotqa/
```

## Run ConditionalQA With OpenAI

```bash
python adapters/run_conditionalqa_vllm.py \
  --provider openai \
  --llm-model gpt-3.5-turbo-0125 \
  --embedding-model text-embedding-3-small \
  --ingest \
  --limit 10
```

## Run ConditionalQA With A Local Model

```bash
python adapters/run_conditionalqa_vllm.py \
  --provider openai-compatible \
  --llm-model local-qa-model \
  --llm-base-url http://127.0.0.1:8050/v1 \
  --llm-api-key not_needed \
  --embedding-model text-embedding-3-small \
  --embedding-base-url https://api.openai.com/v1 \
  --embedding-api-key "$OPENAI_API_KEY" \
  --ingest \
  --limit 10
```

## Evaluate ConditionalQA

```bash
python adapters/eval_conditionalqa_bridge.py \
  --input adapter_runs/conditionalqa/predictions.jsonl
```

## Run HotpotQA With OpenAI

```bash
python adapters/run_hotpotqa_vllm.py \
  --provider openai \
  --llm-model gpt-3.5-turbo-0125 \
  --embedding-model text-embedding-3-small \
  --ingest \
  --limit 10
```

## Run HotpotQA With A Local Model

```bash
python adapters/run_hotpotqa_vllm.py \
  --provider openai-compatible \
  --llm-model local-qa-model \
  --llm-base-url http://127.0.0.1:8051/v1 \
  --llm-api-key not_needed \
  --embedding-model text-embedding-3-small \
  --embedding-base-url https://api.openai.com/v1 \
  --embedding-api-key "$OPENAI_API_KEY" \
  --graph-storage NetworkXStorage \
  --ingest \
  --limit 10
```

Ingest only:

```bash
python adapters/run_hotpotqa_vllm.py \
  --provider openai-compatible \
  --llm-model local-qa-model \
  --llm-base-url http://127.0.0.1:8051/v1 \
  --llm-api-key not_needed \
  --embedding-model text-embedding-3-small \
  --embedding-base-url https://api.openai.com/v1 \
  --embedding-api-key "$OPENAI_API_KEY" \
  --graph-storage NetworkXStorage \
  --ingest \
  --ingest-only
```

## Evaluate HotpotQA

```bash
python adapters/eval_hotpotqa_bridge.py \
  --input adapter_runs/hotpotqa/predictions.jsonl
```

## Output Paths

```text
adapter_data/<dataset>/
workspaces/<dataset>/
adapter_runs/<dataset>/
```

Evaluation files:

```text
normalized_output.jsonl
eval_predictions.jsonl
eval_predictions.json
results.json
```

## Main Options

```text
--provider openai|openai-compatible
--llm-model MODEL
--llm-base-url URL
--llm-api-key KEY
--embedding-model MODEL
--embedding-base-url URL
--embedding-api-key KEY
--embedding-dim N
--working-dir PATH
--output-path PATH
--limit N
--ingest
--ingest-only
--max-total-tokens N
--structured-retries N
--no-structured-output
--json-enforcement
--graph-storage NetworkXStorage|Neo4JStorage
--no-llm-cache
```
