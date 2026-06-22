#!/bin/bash

# Activate environment if needed
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate llama-test

echo "🔍 Running HybridContextQA on HotpotQA..."

python ../../core_utils/retrieve_context.py \
  --input ../../data/hotpot_dev_distractor_500.json \
  --output ../../outputs/hybrid_context.jsonl \
  --framework openai \
  --model gpt-4o \
  --vector-storage ./storage/dev_wiki_articles_500_neo4j_faiss_kg_v3 \
  --kg-storage ./storage/dev_wiki_articles_500_neo4j_faiss_kg_v2 \
  --similarity-top-k 3 \
  --kg-top-k 30

python ../../core_utils/run_qa_from_context.py \
  --input ../../outputs/hybrid_context.jsonl \
  --output ../../outputs/hybrid_answers.jsonl \
  --gold_data ../../data/hotpot_dev_distractor_500.json \
  --framework openai \
  --model gpt-4o \
  --context hybrid
