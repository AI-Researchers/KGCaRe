#!/usr/bin/env bash
set -euo pipefail

HYBRIDRAG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "${HYBRIDRAG_DIR}")"
cd "${WORKSPACE_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-full}"
KG_MODEL="${KG_MODEL:-gpt-4.1-mini}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
INDEX_DIR="${INDEX_DIR:-HybridRAG/indexes/conditionalqa/dev-gpt-4-1-mini-shared}"

"${PYTHON_BIN}" HybridRAG/prepare_conditionalqa.py

if [[ ! -f "${INDEX_DIR}/index_manifest.json" ]]; then
  "${PYTHON_BIN}" HybridRAG/build_index.py \
    --index-dir "${INDEX_DIR}" \
    --kg-model "${KG_MODEL}" \
    --embedding-model "${EMBEDDING_MODEL}"
fi

run_and_eval() {
  local provider="$1"
  local model="$2"
  local base_url="$3"

  local extra_args=()
  if [[ -n "${base_url}" ]]; then
    extra_args+=(--llm-base-url "${base_url}")
  fi

  "${PYTHON_BIN}" HybridRAG/run_hybrid.py \
    --provider "${provider}" \
    --llm-model "${model}" \
    --run-name "${RUN_NAME}" \
    --index-dir "${INDEX_DIR}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --overwrite \
    "${extra_args[@]}"

  local provider_slug
  provider_slug="$(printf "%s" "${provider}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g')"
  local model_slug
  model_slug="$(printf "%s" "${model}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g')"
  "${PYTHON_BIN}" HybridRAG/eval_conditionalqa.py \
    --input "HybridRAG/adapter_runs/conditionalqa/${provider_slug}/${model_slug}/${RUN_NAME}/predictions.jsonl"
}

run_and_eval "openai" "gpt-4.1-nano" ""
run_and_eval "openai" "gpt-4.1-mini" ""

MISTRAL_BASE_URL="${MISTRAL_BASE_URL:-${VLLM_BASE_URL:-http://127.0.0.1:8050/v1}}"
MIXTRAL_BASE_URL="${MIXTRAL_BASE_URL:-${VLLM_BASE_URL:-http://127.0.0.1:8051/v1}}"

run_and_eval "vllm" "mistralai/Mistral-7B-Instruct-v0.3" "${MISTRAL_BASE_URL}"
run_and_eval "vllm" "mistralai/Mixtral-8x7B-Instruct-v0.1" "${MIXTRAL_BASE_URL}"
