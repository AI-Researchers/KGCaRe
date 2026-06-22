#!/bin/bash

echo "🔍 Running KGCaRe on HotpotQA..."

python ../../core_utils/retrieve_context_KGCaRe.py \
  --input ../../data/hotpot_dev_distractor_500.json \
  --output ../../outputs/kg_context.jsonl \
  --framework openai \
  --model gpt-4o \
  --vector-storage ./storage/dev_wiki_articles_500_neo4j_faiss_kg_v3 \
  --kg-storage ./storage/dev_wiki_articles_500_neo4j_faiss_kg_v2 \
  --similarity-top-k 0 \
  --kg-top-k 30

python ../../core_utils/run_qa_from_context.py \
  --input ../../outputs/kg_context.jsonl \
  --output ../../outputs/kg_answers.jsonl \
  --gold_data ../../data/hotpot_dev_distractor_500.json \
  --framework openai \
  --model gpt-4o \
  --context kg
