# KGCaRe

This repository contains code for running KGCaRe and baseline adapters on the included datasets.

## Folders

```text
KGCaRe/                KGCaRe implementation
HybridRAG/             HybridRAG baseline adapter
lightRAG/adapters/     LightRAG adapter scripts
HippoRAG/adapters/     HippoRAG adapter scripts
core_utils/            Shared utility and evaluation code
data/                  ConditionalQA and sampled HotpotQA files
```

Run instructions:

```text
KGCaRe/README.md
HybridRAG/README.md
lightRAG/adapters/README.md
HippoRAG/adapters/README.md
```

## Environment Variables

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1

export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=...
export NEO4J_DATABASE=neo4j
```

For local OpenAI-compatible servers:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=not_needed
```

## Main KGCaRe Commands

Install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r KGCaRe/requirements.txt
```

Build a ConditionalQA index:

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

Run QA:

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

Evaluate:

```bash
python KGCaRe/eval_conditionalqa.py \
  --run-dir KGCaRe/adapter_runs/conditionalqa/openai/gpt-3.5-turbo-0125/full
```
