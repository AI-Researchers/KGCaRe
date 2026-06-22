#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/clean/checkout" >&2
  exit 2
fi

target="$1"
source_root="$(pwd -P)"
target_root="$(cd "$target" && pwd -P)"

if [[ ! -d "$target/.git" ]]; then
  echo "Target must be an existing Git checkout: $target" >&2
  exit 2
fi

if [[ "$target_root" == "$source_root" ]]; then
  echo "Refusing to prepare the current working tree as the publish target." >&2
  exit 2
fi

rsync -av --delete --delete-excluded \
  --filter 'P .git/' \
  --filter 'P .git/***' \
  --exclude '.git' \
  --exclude '.agents' \
  --exclude '.codex' \
  --exclude '.deepeval' \
  --exclude '.pytest_cache' \
  --exclude '.venv' \
  --exclude 'vllm-qwen36' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '*.log' \
  --exclude '*.tmp' \
  --exclude '*.bak' \
  --exclude '*.zip' \
  --exclude 'Neo4j-*.txt' \
  --exclude 'workspace_docs' \
  --exclude 'experiments_report.md' \
  --exclude 'HybridContextQ[A]' \
  --exclude 'docs/papers' \
  --exclude 'data/*.pdf' \
  --exclude 'data/*.py' \
  --exclude 'outputs' \
  --exclude 'outputs_final' \
  --exclude 'storage' \
  --exclude '*/adapter_runs' \
  --exclude '*/outputs' \
  --exclude '*/indexes' \
  --exclude '*/workspaces' \
  --exclude '*/adapter_data' \
  --exclude '*/lightrag_cache' \
  --exclude '*/llm_cache' \
  --exclude 'lightRAG/adapters/...' \
  --exclude 'HippoRAG/reproduce/dataset' \
  --exclude 'HippoRAG/demo*.py' \
  --exclude 'HippoRAG/main*.py' \
  --exclude 'HippoRAG/tests*.py' \
  --exclude 'HippoRAG/test_*.py' \
  --exclude 'HippoRAG/images' \
  --exclude 'HippoRAG/reproduce' \
  --exclude 'lightRAG/examples' \
  --exclude 'lightRAG/reproduce' \
  --exclude 'lightRAG/docs' \
  --exclude 'lightRAG/tests' \
  --exclude 'lightRAG/k8s-deploy' \
  --exclude 'lightRAG/lightrag_webui' \
  --exclude 'lightRAG/README.assets' \
  --exclude 'lightRAG/assets' \
  --exclude 'lightRAG/.github' \
  --exclude 'lightRAG/.clinerules' \
  --exclude 'lightRAG/scripts' \
  --exclude 'lightRAG/lightrag/api' \
  --exclude 'lightRAG/lightrag/evaluation' \
  --exclude 'lightRAG/lightrag/tools' \
  --exclude 'lightRAG/docker*' \
  --exclude 'lightRAG/env.*' \
  --exclude 'lightRAG/README-zh.md' \
  --exclude '*/.git' \
  ./ "$target"/

echo "Prepared publish artifact in $target"
