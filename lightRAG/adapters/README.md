# LightRAG Local Adapters

This folder contains workspace-specific adapter code for running `LightRAG` against the datasets and evaluation flows in the parent repo.

## Purpose
- Keep upstream `LightRAG` code isolated.
- Prepare `ConditionalQA` and `HotpotQA` artifacts in stable JSONL formats.
- Add local run scaffolds for `vLLM`-served `Mistral` / `Mixtral` models.
- Preserve apples-to-apples evaluation by reusing the parent repo's existing evaluators.

## Files
- `common.py`: shared path resolution and JSON/JSONL helpers.
- `output_schemas.py`: initial Pydantic schemas for structured-answer validation.
- `conditionalqa_adapter.py`: prepares `ConditionalQA` corpus and query JSONL files.
- `hotpotqa_adapter.py`: prepares `HotpotQA` corpus and query JSONL files.
- `run_conditionalqa_vllm.py`: initial `LightRAG` runner scaffold for `ConditionalQA`.
- `run_hotpotqa_vllm.py`: initial `LightRAG` runner scaffold for `HotpotQA`.
- `eval_conditionalqa_bridge.py`: normalize raw responses and score with existing `ConditionalQA` evaluator.
- `eval_hotpotqa_bridge.py`: normalize raw responses and compute Hotpot-style EM/F1 summary.

## Quick Start
From the `lightRAG/` directory:

```bash
python adapters/conditionalqa_adapter.py
python adapters/hotpotqa_adapter.py
```

This writes prepared artifacts to:
- `lightRAG/adapter_data/conditionalqa/`
- `lightRAG/adapter_data/hotpotqa/`

Run baseline query scaffolds:

```bash
python adapters/run_conditionalqa_vllm.py --limit 5
python adapters/run_hotpotqa_vllm.py --limit 5
```

Structured output controls:

```bash
# ConditionalQA: structured output enabled by default
python adapters/run_conditionalqa_vllm.py --limit 5
python adapters/run_conditionalqa_vllm.py --no-structured-output --limit 5

# HotpotQA: structured output optional
python adapters/run_hotpotqa_vllm.py --structured-output --limit 5
```

Run with OpenAI models (no vLLM required):

```bash
export OPENAI_API_KEY=your_key_here

# ConditionalQA smoke test with GPT-4.1-nano + OpenAI embeddings
python adapters/run_conditionalqa_vllm.py \
	--provider openai \
	--llm-model gpt-3.5-turbo-0125 \
	--embedding-model text-embedding-3-small \
	--ingest \
	--limit 1

# HotpotQA smoke test
python adapters/run_hotpotqa_vllm.py \
	--provider openai \
	--llm-model gpt-3.5-turbo-0125 \
	--embedding-model text-embedding-3-small \
	--ingest \
	--limit 1
```

Bridge raw outputs into evaluation formats:

```bash
python adapters/eval_conditionalqa_bridge.py
python adapters/eval_hotpotqa_bridge.py
```

## Current Status
This is the first integration scaffold.

Implemented:
- dataset preparation
- shared path helpers
- initial `vLLM` query runners
- structured-output schema placeholders
- ConditionalQA/HotpotQA evaluation bridge scripts

Not finished yet:
- robust retry/validation loop for structured outputs
- full end-to-end invocation of all parent repo evaluation scripts from one command
- run manifests and experiment matrix automation
