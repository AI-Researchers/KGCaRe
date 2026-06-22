# HippoRAG Local Adapters

This directory contains workspace-specific scripts to run HippoRAG on:
- ConditionalQA
- HotpotQA

The adapters are designed to keep output schemas compatible with existing KGCaRe evaluation flows.

## 1) Prepare adapter data

From `HippoRAG/`:

```bash
python adapters/conditionalqa_adapter.py
python adapters/hotpotqa_adapter.py
```

## 2) Run smoke experiments

OpenAI models:

```bash
python adapters/run_conditionalqa_openai.py --llm-model gpt-4.1-nano --limit 20 --run-name smoke

python adapters/run_hotpotqa_openai.py --llm-model gpt-4.1-nano --limit 20 --run-name smoke
```

vLLM models on one OpenAI-compatible endpoint:

```bash
python adapters/run_conditionalqa_vllm.py --llm-model mistralai/Mistral-7B-Instruct-v0.3 --llm-base-url http://127.0.0.1:8051/v1 --limit 20 --run-name smoke

python adapters/run_hotpotqa_vllm.py --llm-model mistralai/Mistral-7B-Instruct-v0.3 --llm-base-url http://127.0.0.1:8051/v1 --limit 20 --run-name smoke
```

Notes:
- For first run per dataset/model, add `--ingest` to build index artifacts.
- Embeddings default to `text-embedding-3-small` using OpenAI-compatible API. Override with:
  - `--embedding-model`
  - `--embedding-base-url`

## 3) Bridge outputs into KGCaRe-compatible evaluation format

ConditionalQA:

```bash
python adapters/eval_conditionalqa_bridge.py --input <path-to-predictions.jsonl>
```

HotpotQA:

```bash
python adapters/eval_hotpotqa_bridge.py --input <path-to-predictions.jsonl>
```

## 4) Full smoke matrix helper

```bash
bash adapters/run_matrix_smoke.sh
```

Artifacts are written under:
- `HippoRAG/adapter_data/`
- `HippoRAG/adapter_runs/`
