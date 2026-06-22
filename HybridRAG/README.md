# HybridRAG Baseline Adapter

`HybridRAG/` contains the paper artifact baseline that combines vector retrieval with graph-triple retrieval. It is not the proposed KGCaRe method. It is retained so the comparison setup can be reproduced from the same repository.

The implementation builds a local FAISS vector index, extracts triples into a NetworkX-style graph representation, retrieves both passage and graph context for each query, and sends the retrieved context to a structured QA prompt.

## Role In The Artifact

HybridRAG is useful as a controlled comparison because it uses both dense retrieval and graph triples, but it does not perform KGCaRe's Neo4j traversal with iterative pruning, reasoning, and clue-entity expansion. In other words, it has a graph channel, but the graph channel is simpler and less symbolic than KGCaRe's traversal module.

## Implementation Map

```text
HybridRAG/prepare_conditionalqa.py       Prepare ConditionalQA corpus/query JSONL
HybridRAG/prepare_hotpotqa.py            Prepare HotpotQA corpus/query JSONL
HybridRAG/build_index.py                 Build ConditionalQA vector and graph-triple index
HybridRAG/build_hotpotqa_index.py        Build HotpotQA vector and graph-triple index
HybridRAG/run_hybrid.py                  Run ConditionalQA HybridRAG QA
HybridRAG/run_hotpotqa.py                Run HotpotQA HybridRAG QA
HybridRAG/eval_conditionalqa.py          Evaluate ConditionalQA predictions
HybridRAG/eval_hotpotqa.py               Evaluate HotpotQA predictions
HybridRAG/run_matrix.sh                  Run the ConditionalQA model matrix
HybridRAG/hybridrag/data.py              Dataset paths, output paths, JSONL helpers
HybridRAG/hybridrag/indexing.py          FAISS index and graph-triple index construction
HybridRAG/hybridrag/retrieval.py         Vector retrieval and graph context selection
HybridRAG/hybridrag/prompts.py           KG extraction and QA prompts
HybridRAG/hybridrag/schemas.py           Structured answer schemas and normalization
```

## Data

ConditionalQA defaults:

- Documents: `data/docs_dev/*.txt`
- Questions: `data/dev.json`
- Prepared adapter files: `HybridRAG/adapter_data/conditionalqa/`

HotpotQA defaults:

- Documents: `data/wiki_articles_supported_500/*.txt`
- Questions: `data/stratified_hotpotqa_500sample_with_tag.json`
- Prepared adapter files: `HybridRAG/adapter_data/hotpotqa/`

Generated adapter data, indexes, and run outputs are ignored by Git.

## Environment

Install the baseline dependencies separately from KGCaRe:

```bash
python -m pip install -r HybridRAG/requirements.txt
```

Set OpenAI credentials for OpenAI models and embeddings:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
```

For local OpenAI-compatible models, use:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=not_needed
```

## Model Defaults

- KG construction: `gpt-4o-2024-08-06`
- OpenAI QA models used in the matrix: `gpt-3.5-turbo-0125`, `gpt-4o-2024-08-06`
- Embeddings: `text-embedding-3-small`
- Local model examples: `mistralai/Mistral-7B-Instruct-v0.3`, `mistralai/Mixtral-8x7B-Instruct-v0.1`

## ConditionalQA Workflow

Prepare adapter data:

```bash
python HybridRAG/prepare_conditionalqa.py
```

Build the shared ConditionalQA index:

```bash
python HybridRAG/build_index.py \
  --kg-model gpt-4o-2024-08-06 \
  --embedding-model text-embedding-3-small
```

Run QA:

```bash
python HybridRAG/run_hybrid.py \
  --provider openai \
  --llm-model gpt-3.5-turbo-0125 \
  --run-name full
```

Evaluate:

```bash
python HybridRAG/eval_conditionalqa.py \
  --input HybridRAG/adapter_runs/conditionalqa/openai/gpt-3.5-turbo-0125/full/predictions.jsonl
```

Run a smoke test:

```bash
python HybridRAG/run_hybrid.py \
  --provider openai \
  --llm-model gpt-3.5-turbo-0125 \
  --run-name smoke \
  --limit 10 \
  --overwrite
```

## HotpotQA Workflow

Prepare adapter data:

```bash
python HybridRAG/prepare_hotpotqa.py
```

Build the HotpotQA index. The artifact's HotpotQA index can be built with an OpenAI-compatible KG model:

```bash
python HybridRAG/build_hotpotqa_index.py \
  --kg-provider vllm \
  --kg-model Qwen3.6-27B \
  --kg-base-url http://127.0.0.1:8000/v1 \
  --kg-api-key not_needed \
  --embedding-model text-embedding-3-small \
  --structured-method json
```

Run QA:

```bash
python HybridRAG/run_hotpotqa.py \
  --provider openai \
  --llm-model gpt-3.5-turbo-0125 \
  --run-name full
```

Evaluate:

```bash
python HybridRAG/eval_hotpotqa.py \
  --input HybridRAG/adapter_runs/hotpotqa/openai/gpt-3.5-turbo-0125/full/predictions.jsonl
```

## Retrieval Settings

The main QA scripts expose comparable retrieval controls:

- `--vector-fetch-k`: number of vector candidates initially fetched.
- `--vector-context-k`: number of vector chunks passed into the final prompt.
- `--graph-seed-k`: number of vector hits used to seed graph context.
- `--graph-depth`: graph expansion depth.
- `--graph-context-k`: number of graph triples passed into the final prompt.
- `--structured-method`: structured generation mode, usually `auto`.
- `--structured-retries`: number of parse retry attempts.
- `--on-parse-error`: whether to raise or write a fallback answer.

The default prompt places vector context before graph-triple context. This is different from KGCaRe, where KG traversal can iteratively prune triples and request clue entities before answer generation.

## Output Layout

ConditionalQA:

```text
HybridRAG/adapter_runs/conditionalqa/<provider>/<model-slug>/<run-name>/
  predictions.jsonl
  run_config.json
  results.json
```

HotpotQA:

```text
HybridRAG/adapter_runs/hotpotqa/<provider>/<model-slug>/<run-name>/
  predictions.jsonl
  run_config.json
  results.json
```

Index directories:

```text
HybridRAG/indexes/conditionalqa/dev-gpt-4o-2024-08-06-shared/
HybridRAG/indexes/hotpotqa/stratified-qwen-shared/
```

These generated paths are not committed.

## Full ConditionalQA Matrix

```bash
bash HybridRAG/run_matrix.sh
```

Optional environment overrides:

```bash
export PYTHON_BIN=python
export RUN_NAME=full
export KG_MODEL=gpt-4o-2024-08-06
export MISTRAL_BASE_URL=http://127.0.0.1:8050/v1
export MIXTRAL_BASE_URL=http://127.0.0.1:8051/v1
```

## Notes For Paper Reviewers

- HybridRAG uses graph triples, but it does not use Neo4j traversal.
- The graph context is selected as a retrieval feature, not as an iterative reasoning path.
- The KG extractor is a two-step prompt process: compact factual notes, then validated triples.
- Outputs are normalized into the same broad evaluation format used by the rest of the artifact.
- For fair comparisons, use the same dataset split, embedding model, and QA model matrix as KGCaRe.
