# KGCaRe Paper Artifact

This repository contains the code artifact for **KGCaRe**, a knowledge-graph-augmented retrieval and question answering pipeline for complex conditional QA. The main implementation is in `KGCaRe/`. Baseline adapters retained for comparison are in `HybridRAG/`, `lightRAG/adapters/`, and `HippoRAG/adapters/`.

KGCaRe builds a document-level knowledge graph with LLM-extracted triples, stores the triples in Neo4j, stores normalized text embeddings in FAISS, retrieves graph and vector context separately, and passes the combined context to the answer-generation prompt. The vector channel is a supporting retrieval path: it preserves source-text wording and exact span conditions that may not survive triple extraction.

## Repository Layout

```text
KGCaRe/                Main KGCaRe implementation
HybridRAG/             HybridRAG baseline adapter
lightRAG/adapters/     LightRAG dataset adapters and evaluation bridges
HippoRAG/adapters/     HippoRAG dataset adapters and evaluation bridges
core_utils/            Shared legacy utilities and evaluation helpers
data/                  ConditionalQA and sampled HotpotQA data used by the artifact
docs/                  Method notes and artifact documentation
scripts/               Publishing and cleanup utilities
```

Detailed method-specific instructions are available in:

- `KGCaRe/README.md`
- `HybridRAG/README.md`
- `lightRAG/adapters/README.md`
- `HippoRAG/adapters/README.md`

## Main Models Used

The artifact defaults are aligned with the paper experiments:

- KG construction model: `gpt-4o-2024-08-06`
- OpenAI QA baseline model: `gpt-3.5-turbo-0125`
- Embedding model: `text-embedding-3-small`

Local vLLM/OpenAI-compatible models can be used by passing `--qa-base-url`, `--kg-base-url`, and `--*-api-key not_needed`.

## Setup

Use Python 3.10+ and install the main KGCaRe dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r KGCaRe/requirements.txt
```

Configure API and Neo4j credentials:

```bash
export OPENAI_API_KEY=...
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=...
export NEO4J_DATABASE=neo4j
```

A local Neo4j 5 server is required for KG storage and traversal.

## Build A KGCaRe Index

ConditionalQA:

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

HotpotQA sample:

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

Indexes are written under `KGCaRe/indexes/`. Generated indexes are ignored by Git.

## Run KGCaRe QA

Hybrid KG + vector retrieval:

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

KG-only or vector-only ablations can be run with `--mode kg` or `--mode vector`.

Evaluate ConditionalQA:

```bash
python KGCaRe/eval_conditionalqa.py \
  --run-dir KGCaRe/adapter_runs/conditionalqa/openai/gpt-3.5-turbo-0125/full
```

Evaluate HotpotQA:

```bash
python KGCaRe/eval_hotpotqa.py \
  --run-dir KGCaRe/adapter_runs/hotpotqa/openai/gpt-3.5-turbo-0125/full
```

Run outputs are written under `KGCaRe/adapter_runs/` and are ignored by Git.

## Baseline Adapters

The retained baselines are included for reproducibility and comparison:

- `HybridRAG/`: vector + graph-triple retrieval baseline.
- `lightRAG/adapters/`: dataset adapters and evaluation bridges for LightRAG.
- `HippoRAG/adapters/`: dataset adapters and evaluation bridges for HippoRAG.

Each baseline directory has its own README or adapter scripts. Install baseline-specific dependencies separately when running those experiments.

## Artifact Notes

- The repository intentionally excludes generated indexes, run outputs, local workspaces, caches, virtual environments, and private credentials.
- Neo4j credentials are read from environment variables; no local passwords are stored in the artifact.
- The FAISS vector store uses normalized embeddings with exact inner-product search through `faiss.IndexFlatIP`.
- See `docs/kg_graph_creation_approaches.md` for notes comparing the KG construction behavior of KGCaRe and the retained baselines.
