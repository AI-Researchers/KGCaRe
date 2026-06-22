# README: ConditionalQA - Retrieval-Augmented QA Pipelines

This repository provides two distinct RAG pipelines:

* `HybridContextQA`: Combines vector-based and KG-table-based retrieval.
* `KGCaRe`: Incorporates advanced multi-hop KG reasoning with traversal-based KG retrieval.

Each is tested on the **ConditionalQA** dataset.

---

## 1. Dataset: ConditionalQA : ( https://haitian-sun.github.io/conditionalqa/ )

**Paper**: [https://arxiv.org/abs/2110.06884](https://arxiv.org/abs/2110.06884)

**Dataset GitHub**: [https://github.com/haitian-sun/ConditionalQA](https://github.com/haitian-sun/ConditionalQA)

**Dataset TRAIN**: [https://github.com/haitian-sun/ConditionalQA/blob/master/v1_0/train.json](https://github.com/haitian-sun/ConditionalQA/blob/master/v1_0/train.json)

**Dataset DEV**: [https://github.com/haitian-sun/ConditionalQA/blob/master/v1_0/dev.json](https://github.com/haitian-sun/ConditionalQA/blob/master/v1_0/dev.json)

**Dataset DOCUMENTS**: [https://github.com/haitian-sun/ConditionalQA/blob/master/v1_0/documents.json](https://github.com/haitian-sun/ConditionalQA/blob/master/v1_0/documents.json)

### Structure

* **dev.json** / **train.json** contain fields:

```json
{
    "url": "https://www.gov.uk/housing-benefit",
    "scenario": "I'm 71, and am currently living in rented accommodation with my 64-year-old Civil Partner. I have an existing Housing Benefit claim.",
    "question": "Can I continue to claim Housing Benefit?",
    "not_answerable": false,
    "answers": [
      [
        "yes",
        []
      ]
    ],
    "evidences": [
      "<p>Your existing claim will not be affected if, before 15 May 2019, you:</p>",
      "<li>were getting Housing Benefit</li>",
      "<li>had reached State Pension age</li>",
      "<p>It does not matter if your partner is under State Pension age.</p>"
    ],
    "id": "dev-13"
  }
```

### Document Preparation

The documents (used for retrieval) are saved as HTML-like pages with semantic structures extracted from ConditionalQA guidance. These are stored under:

```bash
./data/docs_dev/  
```

To convert from original JSON documents:

```python
import json

# Load the JSON data
with open("documents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for doc in data:
    title = doc["title"]
    url = doc["url"]
    content = "\n".join(doc["contents"])

    output = f"{title}\n{url}\n{content}"

    # Save to .txt file (sanitize title for filename)
    filename = title.replace(" ", "_").replace("/", "_") + ".txt"
    with open(filename, "w", encoding="utf-8") as out_file:
        out_file.write(output)

    print(f"Saved: {filename}")

```

---

## 2. Setup and Installation

### 2.1 Python Environment

```bash
conda create -n llm-project python=3.10 -y
conda activate llm-project
pip install -r requirements.txt
```

### 2.2 Neo4j Graph Database

Used for storing and traversing the Knowledge Graph.

#### Installation:

```bash
# Download from https://neo4j.com/download/
# Or install via apt (Ubuntu):
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 5' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update && sudo apt install neo4j
```

#### Configuration:

* Set password in: `neo4j.conf`
* Start the server:

```bash
sudo systemctl enable neo4j
sudo systemctl start neo4j
```

* Configure authentication:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password_here
```

---

## 3. LLM + Embedding Settings

You can use:

* **OpenAI** (`gpt-4`, `gpt-3.5-turbo`)
* **Mistral** via **vLLM** or **OpenAI-compatible API**
* **Huggingface** models (like `llama3`) via `ollama`

### API Keys

Set as environment variables:

```bash
export OPENAI_API_KEY=...
export CO_API_KEY=...
```

### Embedding

Default: `BAAI/bge-large-en-v1.5` via Huggingface.

### Reranker Options

* `cohere` (default): Uses Cohere ReRank API
* `llm`: LLM-based reranking using the loaded model
* `none`: No reranking

---

## 4. Pipeline Options

### 4.1 HybridContextQA

Combines:

* **KGTableRetriever** with embedding
* **VectorIndexRetriever**
* Optional reranking

**Path**: `HybridContextQA/conditionalqa/run.py`

Run:

```bash
python HybridContextQA/conditionalqa_run.py \
  --llm_framework openai \
  --llm_model gpt-4o \
  --index hybrid_index \
  --reranker cohere \
  --num_shots 4
```

### 4.2 KGCaRe

Uses:

* **KGRetrieverToGTraversal\_final**: Neo4j multi-hop KG traversal
* **VectorIndexRetriever**
* Supports `multi_prompt=True` for KGIndex construction

**Path**: `KGCaRe/conditionalqa/run.py`

Run:

```bash
python KGCaRe/conditionalqa_run.py \
  --llm_framework openai \
  --llm_model gpt-4o \
  --index hybrid_index \
  --reranker llm \
  --num_shots 4
```

---

## 5. Output Format

Each run creates:

```bash
outputs_final/dev_kg_v3/ConditionalQA_test/
  └── <llm_model>/<index>/shots_<n>/run_<m>/
      ├── output.jsonl           # model predictions
      ├── skipped_ids.json       # failed attempts
      └── results.json           # final aggregated scores
```

### Sample Output Entry

```json
{
  "id": "001",
  "Prompt": "...",
  "Question": "My brother and his wife are in prison for carrying out a large fraud scheme. Their 7 and 8 year old children have been living with me for the last 4 years. I want to become their Special Guardian to look after them permanently How long will it be before I hear back from the court?",
  "Question_Type": "span",
  "answers": [
            [
                "Within 10 days of receiving your application",
                []
            ]
        ],
  "Actual_Answer": [
            [
                "within 10 days",
                []
            ]
        ],
  "Score": {
            "EM": 0.866013071895425,
            "Conditional_EM": 0.866013071895425,
            "F1": 0.7445266842842102,
            "Conditional_F1": 0.7445266842842102
        }
}
```

---

## 6. Notes

* Evaluation supports both EM/F1 and METEOR/BERTScore.
* Can resume failed runs using `--retry_skipped`
* Easy to switch between retrievers via `--index` flag:

  * `kg_index`, `vector_index`, `hybrid_index`, `no_index`

---


# README: HotpotQA - HybridContextQA and KGCaRe Pipelines

This README provides complete setup and usage instructions for running **HybridContextQA** and **KGCaRe** pipelines on the **HotpotQA** dataset.

---

## 📘 1. Dataset: HotpotQA

**Paper**: [https://arxiv.org/pdf/1809.09600.pdf](https://arxiv.org/pdf/1809.09600.pdf)
**Dataset Link**: [https://hotpotqa.github.io/](https://hotpotqa.github.io/)

### ✅ Format

HotpotQA provides multi-hop QA with supporting context paragraphs.

For this pipeline, we only use the **stratified_hotpotqa_500sample_with_tag.json** file:

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "question": "Was Meghan Markle born in the same year as Teresa Palmer?",
  "answer": "No",
  "type": "comparison",
  "level": "medium",
  "supporting_facts": [...],
  "context": [...]
}
```

We format and save these into:

```bash
./data/hotpotqa/dev.json
```

---

## 🗂️ 2. Folder Structure

```
project_root/
|
├── core_utils/                   # Shared scripts (retrieval, QA, utils)
│   ├── retrieve_context.py
│   ├── run_qa_from_context.py
│   └── ...
│
├── HybridContextQA/
│   └── hotpotqa/
│       ├── run.py
│       └── run_hybrid_hotpotqa.sh
│
├── KGCaRe/
│   └── hotpotqa/
│       ├── run.py
│       └── run_kg_hotpotqa.sh
│
├── data/
│   └── hotpotqa/
│       ├── dev.json               # Formatted questions
│       └── docs_dev/             # Parsed document .txt files
│           ├── 0001.txt
│           └── ...
│
├── storage/
│   ├── hotpot_vector_index/      # FAISS vector store
│   └── hotpot_kg_index/          # Neo4j KG index
│
├── outputs_final/
│   └── hotpotqa/
│       ├── hybrid_dev.jsonl
│       ├── hybrid_answers.jsonl
│       ├── kg_dev.jsonl
│       └── kg_answers.jsonl
│
├── requirements.txt
└── README.md
```

---

## ⚙️ 3. Setup Instructions

### 3.1 Python Environment

```bash
conda create -n llm-project python=3.10 -y
conda activate llm-project
pip install -r requirements.txt
```

### 3.2 Neo4j Setup

Used to store and traverse KG triples.

```bash
# Install (Ubuntu)
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 5' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update && sudo apt install neo4j
```

#### Configure and Start

```bash
sudo systemctl enable neo4j
sudo systemctl start neo4j

export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password_here
```

### 3.3 API Keys

```bash
export OPENAI_API_KEY=...
export CO_API_KEY=...
```

---

## 🧠 4. LLM, Embeddings & Reranker

### Supported LLM Frameworks

* `openai`        → `gpt-4o`, `gpt-3.5-turbo`
* `openai_like`   → Mistral etc. via vLLM / Ollama

### Embedding Model

* `BAAI/bge-large-en-v1.5` via HuggingFace

### Reranker Options

* `cohere`: Uses Cohere's ReRank API
* `llm`: Reranking with the main LLM
* `none`: No reranking

---

## 🚀 5. Run Pipelines

### 🔹 HybridContextQA

**Script**: `HybridContextQA/hotpotqa/run_hybrid_hotpotqa.sh`

```bash
chmod +x HybridContextQA/hotpotqa/run_hybrid_hotpotqa.sh
./HybridContextQA/hotpotqa/run_hybrid_hotpotqa.sh
```

Contents of script:

```bash
#!/bin/bash

# Step 1: Retrieve vector + KG context
python ../../core_utils/retrieve_context.py \
  --input ../../data/hotpotqa/dev.json \
  --output ../../outputs_final/hotpotqa/hybrid_dev.jsonl \
  --framework openai \
  --model gpt-4o \
  --vector-storage ../../storage/hotpot_vector_index \
  --kg-storage ../../storage/hotpot_kg_index \
  --similarity-top-k 3 \
  --kg-top-k 30

# Step 2: Answer from context
python ../../core_utils/run_qa_from_context.py \
  --input ../../outputs_final/hotpotqa/hybrid_dev.jsonl \
  --output ../../outputs_final/hotpotqa/hybrid_answers.jsonl \
  --gold_data ../../data/hotpotqa/dev.json \
  --framework openai \
  --model gpt-4o \
  --context hybrid
```

---

### 🔹 KGCaRe

**Script**: `KGCaRe/hotpotqa/run_kg_hotpotqa.sh`

```bash
chmod +x KGCaRe/hotpotqa/run_kg_hotpotqa.sh
./KGCaRe/hotpotqa/run_kg_hotpotqa.sh
```

Contents of script:

```bash
#!/bin/bash

# Step 1: KG traversal-based retrieval
python ../../core_utils/retrieve_context.py \
  --input ../../data/hotpotqa/dev.json \
  --output ../../outputs_final/hotpotqa/kg_dev.jsonl \
  --framework openai \
  --model gpt-4o \
  --vector-storage ../../storage/hotpot_vector_index \
  --kg-storage ../../storage/hotpot_kg_index \
  --similarity-top-k 0 \
  --kg-top-k 30

# Step 2: Answer from context
python ../../core_utils/run_qa_from_context.py \
  --input ../../outputs_final/hotpotqa/kg_dev.jsonl \
  --output ../../outputs_final/hotpotqa/kg_answers.jsonl \
  --gold_data ../../data/hotpotqa/dev.json \
  --framework openai \
  --model gpt-4o \
  --context kg
```

---

## 📥 6. Output Format

Each pipeline saves predictions to `output.jsonl`:

```json
{
  "id": "5a8b57f25542995d1e6f1371",
  "Question": "Was Meghan Markle born in the same year as Teresa Palmer?",
  "answers": "No",
  "Actual_Answer": "No",
  "Score": {
    "EM": 1.0,
    "F1": 1.0,
    "BERT_F1": 1.0
  }
}
```

If a query fails, its ID is logged in `skipped_ids.json`.

---

## 📎 7. Notes

* Works with any OpenAI-compatible LLM or local models via vLLM/Ollama
* You can vary `--context` flag in `run_qa_from_context.py` to use:

  * `vector`
  * `kg`
  * `hybrid`
  * `none`
* Retry failed questions using `--retry_skipped`
* Evaluation metrics: EM, F1, BERTScore (optionally METEOR)

---

