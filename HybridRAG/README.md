# HybridRAG ConditionalQA and HotPotQA

This package implements the paper-style HybridRAG setup for ConditionalQA and
the stratified HotPotQA sample:
vector context is retrieved from FAISS, graph context is retrieved from
NetworkX-backed triples, and final QA receives vector context first followed by
graph-triple context.

The initial experiment uses:

- Dataset: `data/docs_dev/*.txt` and `data/dev.json`
- Embeddings: `OpenAIEmbeddings(model="text-embedding-3-small")`
- Shared KG extractor: `gpt-4.1-mini`
- QA models: `gpt-4.1-nano`, `gpt-4.1-mini`,
  `mistralai/Mistral-7B-Instruct-v0.3`, and
  `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Evaluation: existing `core_utils.conditionalqa_eval`

The HotPotQA adapter uses:

- Dataset: `data/stratified_hotpotqa_500sample_with_tag.json`
- Corpus: `data/wiki_articles_supported_500/*.txt`
- Current prepared size: 500 questions and 999 documents
- Question mix: 475 `span`, 25 `yes/no`
- Evaluation: HotPotQA answer EM/F1 normalization over the stratified sample

## Prompts

The exact prompts used by the implementation live in
`hybridrag/prompts.py`. KG construction is two-step:

1. Refine each document chunk into compact factual notes while preserving exact
   evidence snippets.
2. Extract validated triples from the refined notes.

QA uses a discriminated Pydantic answer schema and then normalizes output into
ConditionalQA's expected `[[answer, [conditions...]]]` format.

## Run

Install dependencies:

```bash
python -m pip install -r HybridRAG/requirements.txt
```

Prepare adapter data:

```bash
python HybridRAG/prepare_conditionalqa.py
```

Build the shared index:

```bash
python HybridRAG/build_index.py \
  --kg-model gpt-4.1-mini \
  --embedding-model text-embedding-3-small
```

Run one QA model:

```bash
python HybridRAG/run_hybrid.py \
  --provider openai \
  --llm-model gpt-4.1-mini \
  --run-name full
```

Evaluate one run:

```bash
python HybridRAG/eval_conditionalqa.py \
  --input HybridRAG/adapter_runs/conditionalqa/openai/gpt-4-1-mini/full/predictions.jsonl
```

Run the full model matrix:

```bash
bash HybridRAG/run_matrix.sh
```

For vLLM/OpenAI-compatible models, set:

```bash
export MISTRAL_BASE_URL=http://127.0.0.1:8050/v1
export MIXTRAL_BASE_URL=http://127.0.0.1:8051/v1
```

## HotPotQA Run

Prepare the stratified HotPotQA adapter data:

```bash
python HybridRAG/prepare_hotpotqa.py
```

Build the one-time HotPotQA index with a Qwen vLLM server:

```bash
python HybridRAG/build_hotpotqa_index.py \
  --kg-provider vllm \
  --kg-model Qwen3.6-27B \
  --kg-base-url http://127.0.0.1:8000/v1 \
  --kg-api-key not_needed \
  --embedding-model text-embedding-3-small \
  --structured-method json
```

Run QA with the shared index:

```bash
python HybridRAG/run_hotpotqa.py \
  --provider openai \
  --llm-model gpt-4.1-mini \
  --run-name full
```

Evaluate:

```bash
python HybridRAG/eval_hotpotqa.py \
  --input HybridRAG/adapter_runs/hotpotqa/openai/gpt-4-1-mini/full/predictions.jsonl
```
