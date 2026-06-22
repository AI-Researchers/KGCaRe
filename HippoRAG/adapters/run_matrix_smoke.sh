#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python adapters/conditionalqa_adapter.py
python adapters/hotpotqa_adapter.py

python adapters/run_conditionalqa_openai.py --llm-model gpt-3.5-turbo-0125 --run-name smoke --limit 20 --ingest
python adapters/run_conditionalqa_vllm.py --llm-model mistralai/Mistral-7B-Instruct-v0.3 --run-name smoke --limit 20 --llm-base-url "${VLLM_BASE_URL:-http://127.0.0.1:8051/v1}" --ingest --qa-top-k 2 --max-new-tokens 256

python adapters/run_hotpotqa_openai.py --llm-model gpt-3.5-turbo-0125 --run-name smoke --limit 20 --ingest
python adapters/run_hotpotqa_vllm.py --llm-model mistralai/Mistral-7B-Instruct-v0.3 --run-name smoke --limit 20 --llm-base-url "${VLLM_BASE_URL:-http://127.0.0.1:8051/v1}" --ingest --qa-top-k 2 --max-new-tokens 256

echo "Smoke matrix completed."
